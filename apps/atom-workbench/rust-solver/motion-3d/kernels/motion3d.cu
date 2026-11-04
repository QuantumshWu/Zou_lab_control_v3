// The 3D motion engine's trajectories on the GPU (src/gpu/mod.rs drives them): engine.rs's steps,
// jumps and loss, model.rs's point, generator, photons, exact response and readout, lib.rs's
// motion windows and sums, rng.rs's streams. The model's tables and descriptors come from
// src/gpu/program.rs; every struct here has its #[repr(C)] twin there.
//
// # Precision
// Single precision where a step's arithmetic is local: the tables, the generator, the propagator,
// the force, the photons. Double precision where it accumulates or decides: the time, the position
// and the momentum with the harmonic flow that turns them, ln‖ψ‖² against ln θ, the random
// deviates and every choice made with them, the kicks, the phases' angles, the exact response's
// diagonal, the motion windows and every sum -- and the readouts of the atom's and the held twin's
// populations, whose rows then sum to one to double precision.
//
// # Layout
// One warp per trajectory, one block per task of the CPU's (lib.rs `TASK`: up to eight consecutive
// trajectories of one group). The warp's lanes share its work: a lane per beam for the fields, a
// lane per matrix entry (e = lane + 32 j, j < SLOTS) for the tables' gathers and the generator, a
// lane per row for the propagator's products and the readout's excited states. The trajectory's
// scalars are uniform over the warp; ψ and the matrices are in its shared memory. A cluster holds
// at most MAXN states: the kernel's bucket (excited states joining clusters would add a larger one,
// with its matrices in shared memory, chosen per solve by the largest cluster).
//
// # Launches
// A launch runs every task of a wave over a range of at most MOST_INTERVALS output intervals
// (src/gpu/mod.rs keeps it short of the display watchdog); the trajectories' states, motion windows
// and samples stay in device memory between launches. Each warp runs its trajectory through the
// launch's intervals on its own, its record of each (the output's, the motion windows closing in it)
// into its slot's records; at the launch's end the block's warps meet, and the records are summed
// per interval over the block in the trajectories' order -- the CPU's order within a task -- into the
// task's sums. The warps wait for the block's slowest once per launch, not once per output.

typedef unsigned int u32;
typedef unsigned long long u64;

#define MAXN 8                          // the most states of a cluster of ground states the kernel holds
#define MAXM 16                         // the most states of a cluster its exact step holds (explicit excited states, beats)
#define EXACT_ROUND 1e-7f               // the exact step's exponentials summed to single precision's round-off
#define SLOTS ((MAXN*MAXN+31)/32)       // a lane's entries of an n × n matrix
#define NG 8                            // ground states
#define NE 24                           // excited states
#define MOST_BEAMS 32
#define MOST_CARRIERS 8
#define MOST_RANK 3
#define MOST_FOCUSED 3
#define MOST_CLASSES 32
#define MOST_LISTED 32
#define WARPS 8                         // trajectories per block: lib.rs `TASK`
// An output's sums (stats.rs `Sums`; src/gpu/mod.rs `sums_of`): held, profile, doppler, the 32
// populations; the held twin's 32 populations, Σ_k c_k² and Σ_k c_k p_k; the folded excited population
// (η²); per lab axis the motion's Σκ, Σκ², Σr, Σr²; per lab axis the harmonic control's Σc, Σc², Σc κ².
#define TWIN 35                         // where the held twin's sums start
#define FOLDED 69                       // where the folded excited population's sum is
#define POINT_RECORD 70                 // the part taken at the output time
#define MOTION 70                       // where the motion windows' means start
#define HARMONIC 82                     // where the harmonic control's sums start
#define RECORD 91
#define CLOSE 8                         // a closing window's record: held, ⟨κ⟩, ⟨κ²⟩, ⟨r⟩, ⟨r²⟩, the harmonic control's c, c², c ⟨κ²⟩
#define SLOT_RECORD (POINT_RECORD+3*2*CLOSE)   // a trajectory's record of an interval: the point's, per lab axis two closing windows' 
#define MOST_INTERVALS 64               // the most output intervals of a launch (the records' room)
#define FULL 0xffffffffu
#define TAU 6.283185307179586
#define PURPOSE_JUMP 3u                 // rng.rs `Purpose::Jump`

// Profiling (examples/profile.rs compiles the kernels with -DPROFILE): the cycles each warp spends in
// the parts of its work, and the advances and jumps it makes, per trajectory into Run::profile.
// Without PROFILE the spans are nothing.
enum Span{SPAN_POINT,SPAN_GENERATOR,SPAN_EXPONENTIAL,SPAN_EVOLVE,SPAN_JUMP,SPAN_EXACT,SPAN_PHOTONS,SPAN_ABSORPTION,SPAN_LOSS,SPAN_STEP,SPAN_READOUT,SPAN_WINDOWS,
    SPAN_SUMS,SPAN_TWIN,SPAN_TOTAL,COUNT_ADVANCES,COUNT_JUMPS,COUNT_EXACT,SPANS};
#ifdef PROFILE
#define SPAN_BEGIN(name) const long long name=clock64()
#define SPAN_END(span,name) do{if(lane==0)W.profile[span]+=clock64()-name;}while(0)
#define COUNT(span) do{if(lane==0)W.profile[span]++;}while(0)
#else
#define SPAN_BEGIN(name) do{}while(0)
#define SPAN_END(span,name) do{}while(0)
#define COUNT(span) do{}while(0)
#endif

// ---------------------------------------------------------------------------------------------
// Complex single precision.
__device__ __forceinline__ float2 operator+(float2 a,float2 b){return make_float2(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ float2 operator-(float2 a,float2 b){return make_float2(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ float2 operator*(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ __forceinline__ float2 operator*(float s,float2 a){return make_float2(s*a.x,s*a.y);}
__device__ __forceinline__ void operator+=(float2& a,float2 b){a.x+=b.x;a.y+=b.y;}
__device__ __forceinline__ void operator-=(float2& a,float2 b){a.x-=b.x;a.y-=b.y;}
__device__ __forceinline__ float2 conj(float2 a){return make_float2(a.x,-a.y);}
__device__ __forceinline__ float norm2(float2 a){return a.x*a.x+a.y*a.y;}
__device__ __forceinline__ float2 zero2(){return make_float2(0.f,0.f);}
__device__ __forceinline__ float2 cdiv(float2 a,float2 b){const float d=1.f/norm2(b);return d*(a*conj(b));}
__device__ __forceinline__ float2 cexp(float2 z){float s,c;sincosf(z.y,&s,&c);const float e=expf(z.x);return make_float2(e*c,e*s);}
// e^{iθ}: θ in double precision, reduced to [−π, π] before the single-precision sine and cosine.
__device__ __forceinline__ float2 cis(double theta){const double x=theta-TAU*rint(theta*(1./TAU));float s,c;sincosf((float)x,&s,&c);return make_float2(c,s);}
// A photon's or a coupling's residual phase e^{iω̃t}: one for a static one (most), with no double precision.
__device__ __forceinline__ float2 residual(double w,double t){return w==0.?make_float2(1.f,0.f):cis(w*t);}

// Complex double precision: the readouts.
__device__ __forceinline__ double2 d2(float2 a){return make_double2((double)a.x,(double)a.y);}
__device__ __forceinline__ double2 dmul(double2 a,double2 b){return make_double2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ __forceinline__ double dnorm2(double2 a){return a.x*a.x+a.y*a.y;}

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(FULL,v,o);
    return v;
}
__device__ __forceinline__ float2 warp_sum(float2 v){return make_float2(warp_sum(v.x),warp_sum(v.y));}
__device__ __forceinline__ double warp_sum(double v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(FULL,v,o);
    return v;
}
__device__ __forceinline__ float warp_max(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(FULL,v,o));
    return v;
}
__device__ __forceinline__ int floordiv(int a,int b){return a>=0?a/b:-((-a+b-1)/b);}

// ---------------------------------------------------------------------------------------------
// Random streams (rng.rs): Philox4x32-10 at the counter (block, event, trajectory) under the key
// (seed ^ purpose·W0, seed >> 32), the words in order, doubles from the high 53 bits of two.
__device__ __forceinline__ uint4 philox(uint4 c,uint2 k){
    #pragma unroll
    for(int round=0;round<10;round++){
        if(round>0){k.x+=0x9E3779B9u;k.y+=0xBB67AE85u;}
        const u32 hi0=__umulhi(0xD2511F53u,c.x),lo0=0xD2511F53u*c.x,hi1=__umulhi(0xCD9E8D57u,c.z),lo1=0xCD9E8D57u*c.z;
        c=make_uint4(hi1^c.y^k.x,lo1,hi0^c.w^k.y,lo0);
    }
    return c;
}
struct Rng{
    uint2 key;uint4 counter,block;int used;
    __device__ Rng(u64 seed,u32 purpose,u64 trajectory,u32 event){
        key=make_uint2((u32)seed^(purpose*0x9E3779B9u),(u32)(seed>>32));counter=make_uint4(0u,event,(u32)trajectory,(u32)(trajectory>>32));used=4;
    }
    __device__ u32 next_u32(){
        if(used==4){block=philox(counter,key);counter.x++;used=0;}
        const u32 x=used==0?block.x:used==1?block.y:used==2?block.z:block.w;used++;return x;
    }
    __device__ u64 next_u64(){const u64 hi=next_u32();return (hi<<32)|next_u32();}
    // Uniform in [0, 1); in (0, 1]; a unit vector uniform on the sphere.
    __device__ double uniform(){return (double)(next_u64()>>11)*(1./9007199254740992.);}
    __device__ double open_uniform(){return 1.-uniform();}
    __device__ void direction(double d[3]){
        const double z=2.*uniform()-1.;const double phi=TAU*uniform();const double rho=sqrt(fmax(1.-z*z,0.));double s,c;sincos(phi,&s,&c);
        d[0]=rho*c;d[1]=rho*s;d[2]=z;
    }
};

// ---------------------------------------------------------------------------------------------
// The model (src/gpu/program.rs builds it from model.rs and ground.rs).

// A beam: its geometry, its kinetic shift, and its roles.
struct Beam{
    float direction[3],k,wave[3];       // unit direction, |k| (rad/um), k n
    float inv_w2,inv_rayleigh;          // 1/w² and 1/z_R of a Gaussian beam; 0 for a plane wave
    float recoil;                       // ħk²/2m (rad/us): the kinetic shift at rest
    float inv_pole;                     // 1/σ_c of its carrier (model.rs `doppler_ratio`); 0 when not near
    int trap;                           // one of the trap's beams (model.rs `trap_beams`): off far from the trap
    int carrier,near;                   // its carrier and its place among the near beams; −1 when not near
    u32 coupled;                        // the ground columns its coupling reaches (bit b)
    float coefficients[MOST_RANK][2];   // c_lk on its carrier's basis
};
struct Carrier{int rank,beams,first,pad;double detuning,wave;};   // its beams: beam_list[first, first + beams); its photons' wave number
// A unit of the light's static shifts (ground.rs `Unit`): its beams unit_beams[first, first + count)
// with their weights γ_l (unit_weights), N, its intensity at the focus s(0) and its weight when taken
// at the focus (model.rs `profile_error`; −1 otherwise). Its intensity s = |Σ_l a_l γ_l|²/N.
struct Unit{int first,count;float norm,focus,profile;int pad;};
// A cluster: its tables' place (node 0 at tables[table], `size` entries per node) and slices, its
// groups, classes and coherent groups, its carriers' photon tables (carrier_tables[carriers + 3c +
// (0 J, 1 J2, 2 T)]), its fourth-order tables (fourths[fourth], −1: folded at second order), its
// states' frame frequencies (frames[frames + x]), where its places' lists of its sparse groups'
// entries start (item_first[positions + e] for the place e = row·n + col), its sparse groups' residual
// beats (residuals[beats, beats + nbeats)) and its states' ground positions. For its exact step (`exact`:
// explicit excited states or beats): its ground states n0, its components' parts (1 + its beats, the beats
// omegas[omegas, omegas + parts − 1) and their period after them, 0 without one), its explicit states'
// terms terms[terms, terms + nterms) and dressed photons (the tables at `emission`), its explicit states'
// excited positions (excited_list[excited]), its groups' parts (group_parts[group_parts + g]) and its
// sparse groups' beats' (residual_parts[residual_parts + b]). The near light's shifts ΔE_a^d its exact
// responses' dressing takes (`exact_responses`): bit d·NG + a.
struct Cluster{int n,size,table,h0,groups,ngroups,classes,nclasses,coherent,ncoherent,carriers,fourth,frames,positions,beats,nbeats,
    n0,exact,parts,omegas,terms,nterms,emission,excited,group_parts,residual_parts;int states[MAXM];u64 shifts;};
// A cluster's fourth-order tables (model.rs `FourthLayout`, # Fourth order): their places in its node
// slice per carrier (−1: none), the carriers that drive it carriers[0, on). H^(4)'s quartic part per
// pair (c, d), r_c² r_d² matrices n × n of the monomials E^c*_k1 E^c_k2 E^d*_k3 E^d_k4; per carrier
// ½[R†R, H_g], R†R and YW, r_c² matrices n × n of E^c*_k1 E^c_k2; for the jumps per pair (c, e)
// C_a G_b(ω_c) T^e_k[:, b′] (r_e × 3 axes × ng rows × ng b′ × n), per pair (d, e) P^{de} (r_d r_e × ng × n),
// per carrier R†R on every ground state (r_c² × ng × n) and C_a T^c_k (r_c × 3 × ng × ng).
struct Fourth{int quartic[MOST_CARRIERS][MOST_CARRIERS],quadratic[MOST_CARRIERS],rr[MOST_CARRIERS],yw[MOST_CARRIERS],
    jumps[MOST_CARRIERS][MOST_CARRIERS],p[MOST_CARRIERS][MOST_CARRIERS],rr_all[MOST_CARRIERS],photons[MOST_CARRIERS],on,carriers[MOST_CARRIERS],pad[3];};
// A kept group (model.rs `GroupLayout`): its field pairs pairs[pairs, pairs + npairs), the S and Q
// slices and its entries' count (n² for a group of every entry, row-major; fewer for a sparse one,
// its values per pair in its entries' order).
struct Group{int absorbed,emitted,hermitian,pairs,npairs,s,q,count;double residual;};
// A jump class (ground.rs `Class`): its photons photons[photons, photons + nphotons) by the row they
// land in and their carrier, the carriers they come from (bits), and where each row's photons start
// (class_rows[rows + r], r up to the target's size).
// Its photons from explicit states directs[direct, direct + ndirect).
struct Class{int target,photons,nphotons,carriers,rows,direct,ndirect,pad;double wave_number;};
// A photon: its carrier, its source column and ground row in the jump tables, its row in the target cluster, ω̃ − ω̃₀.
struct Photon{int carrier,col,to,row;double residual;};
struct Coherent{int channels,count;};
struct Channel{int carrier,col;double residual;};
// A term of a cluster's explicit states (model.rs `exact_terms`): a kept drive (kind 0), a decay cross term
// (1), a decay (2) or a dressing's drive (3); its entry, carrier, part and its mirror's part (the negative
// residual's), its tables' place, a drive's U_k.
struct Term{int kind,row,col,carrier,part,mirror,at,pad;float u[MOST_RANK][2];};
// A class's photon from an explicit state (ground.rs `Class::direct`): its place in the cluster, the ground
// state it lands in and that state's row in the target, its residual.
struct Direct{int state,to,row,pad;double residual;};
// A near carrier's unit for the dressing (ground.rs `NearUnit`): its carrier, its place among the focused units
// (−1: taken at its focus intensity), N and s(0); its field's amplitude² s N at the point.
struct NearUnit{int carrier,focused;float norm,focus;};

struct Program{
    const Beam* beams;const int* beam_list;const Carrier* carriers;const Cluster* clusters;const Group* groups;const int2* pairs;
    const Class* classes;const Photon* photons;const Coherent* coherent;const Channel* channels;const int* carrier_tables;
    const float2* tables;
    // The exact response (model.rs `Exact`): K = H_ee − (i/2)D off its diagonal, its base and per
    // focused beam (NE × NE each); the diagonals in double precision, per block [NG ground | NE
    // excited] (ground ground energies E_b, excited Re K_ii); every beam's folded coupling V_l^f (NE ×
    // NG); the emission C_a (3 × NG × NE).
    const float2* excited;const double* diagonal;const float2* couplings;const float2* emission;
    const int* class_rows;
    const Fourth* fourths;const double* frames;
    // The sparse groups' entries by the place they land at: per cluster and place the items
    // items[item_first[positions + e], item_first[positions + e + 1]) in the groups' and pairs' order,
    // each (its S's and its Q's place in the cluster's node slice, its kind, 0): the kind's bits 0–1
    // the pair's emitted field k1, 2–3 its absorbed k, 4–6 the absorbed carrier, 7–9 the emitted, 10
    // a pair standing for two, 11 the adjoint at the mirror place, 12– the beat (`residuals`).
    const int* item_first;const int4* items;const double* residuals;
    const Unit* units;const int* unit_beams;const float2* unit_weights;
    // The exact step's (`Cluster`): the terms, beats, parts, explicit states, direct photons; the groups of
    // nearly degenerate excited states levels[level_first[g], level_first[g + 1]) and 1/√Γ_ii (the transient).
    const Term* terms;const double* omegas;const int* group_parts;const int* residual_parts;const int* excited_list;const Direct* directs;
    const int* levels;const int* level_first;const float* decay_root;
    // The exact response's dressing (`exact_responses`): every beam's kept drives V_l − V_l^f (NE × NG), D (NE × NE),
    // the near units by carrier (near_units[near_first[d], near_first[d + 1])) and their folded fields W^f
    // (unit_couplings, NG columns of NE each), every channel's far channels (d, a) vias[via_first[c·NG + b],
    // via_first[c·NG + b + 1]).
    const float2* kept;const float2* decay;const NearUnit* near_units;const float2* unit_couplings;const int2* vias;const int* via_first;
    int nbeams,ncarriers,nclusters,focused,nodes,fields,nnear,doppler,loss;
    int weights;                        // where a jump's scratch holds the fourth order's weights
    int nunits;
    int exact_at,transient_at,nlevels,rows_at;  // the exact step's, the transient's and the exact responses' rows' room in a trajectory's scratch
    int focused_unit[MOST_FOCUSED],stride[MOST_FOCUSED];   // the focused units (the trap beam's first), the stencil's strides
    int field_carrier[MOST_CARRIERS*MOST_RANK],field_rank[MOST_CARRIERS*MOST_RANK];
    int ground_states[NG],excited_states[NE],ground_cluster[NG],ground_position[NG];
    int near_first[MOST_CARRIERS+1];
    u32 explicit_of[NG],partners[NE];   // per ground state its cluster's explicit states, per excited state its dressing's partners (bits)
    float hm,light2,allowance,truncation;   // ħ/m (um²/us), twice the light shift's bound, the first order's allowance, the exponentials' truncation
    float axes32[3][3],stiffness32[3];      // the axes and the stiffness below, in single precision
    double hm64,dt;
    double axes[3][3],omega[3],stiffness[3];// the trap's principal axes (rows), ω_i (rad/us), ω_i²/(ħ/m)
    double half_flow[3][3];                 // the flow for dt/2 per axis: cos ωτ, (ħ/m) sin ωτ/ω, (m/ħ) ω sin ωτ
    double region;                          // the trap's region the loss takes (engine.rs `Settings::region`), rad/us
    double node_phase,beats,horizon;        // the exact step's node phase, its beats' share and their horizon (engine.rs `Settings`)
};

// A trajectory's state between launches.
struct State{
    double q[3],p[3];                   // position (um) and momentum over ħ (1/um) on the trap's principal axes
    double time,logn,threshold,log_threshold;
    double prefix[15];                  // per lab axis: steps, Σκ, Σκ², Σr, Σr² through the last step (lib.rs `Windows`)
    double q0[3],p0[3];                 // the initial ones: the harmonic control's
    double twin_logn,twin_threshold,twin_log_threshold;   // the held twin's (control.rs)
    float2 psi[MAXM];                   // ψ/‖ψ‖ on its cluster
    float2 twin[MAXN];                  // the held twin's
    float doppler;int cluster,jumps,death;  // death: the step it was lost at, −1 while held
    u32 index;int twin_cluster,twin_jumps;float beats;   // beats: the largest bound per unit time of the beats its exact steps left out
};
// The held twin (control.rs `Twin`, engine.rs `Engine::held`), none when `u` is null: per cluster k
// its half step's propagator `u` and its H_eff `h` (n_k² entries from blocks[k], row-major) and its
// populations' readout (32 coefficients per density entry from 32·blocks[k], control.rs `readout`);
// per class its jump operators `l` (3 × target × cluster from jumps[class]).
struct Held{const float2* u;const float2* h;const float2* l;const double* readout;const int* blocks;const int* jumps;};
// A launch: a wave's tasks over the output intervals [begin, end).
struct Run{
    State* states;double* rings;double* samples;double* sums;float2* scratch;const int* tasks;
    unsigned long long* profile;        // per slot SPANS counters (with PROFILE)
    double* records;                    // per slot MOST_INTERVALS records of SLOT_RECORD: the launch's intervals
    const double* harmonic;             // per output and lab axis the harmonic control's A and B (control.rs `Harmonic`); none: no control
    unsigned long long* costs;          // per slot the held twin's cycles and the launches' (lib.rs `Outcome::seconds`)
    Held held;
    u64 seed;
    int outputs,steps,reach,begin,end;  // the output times, the steps between two, the steps past the last
    int half[3],ring[3],ring_at[3];     // per lab axis: the window's half-width, its ring's length (steps) and place (doubles)
    int ring_stride,scratch_stride,nlisted,pad;
    int listed[MOST_LISTED];
};

// ---------------------------------------------------------------------------------------------
// A warp's shared memory.

// A carrier's field at the point: E_k, ∇E_k (per axis) and E^D_k (model.rs `CarrierField`).
struct Fields{float2 e[MOST_RANK],grad[MOST_RANK][3],doppler[MOST_RANK];};
// Everything at (r, κ) but the internal state (model.rs `Point`).
struct Point{
    float r[3],kappa[3];                // lab frame
    float s[MOST_FOCUSED],grad_s[MOST_FOCUSED][3];
    int first[MOST_FOCUSED],far_first[MOST_FOCUSED];
    float weight[MOST_FOCUSED][4],slope[MOST_FOCUSED][4],far_weight[MOST_FOCUSED][4];
    float2 a[MOST_BEAMS];float shift[MOST_BEAMS];   // every beam's amplitude and kinetic shift w_l
    Fields fields[MOST_CARRIERS];
    float2 off[MOST_CARRIERS][MOST_RANK];          // E_k with the trap's light off (model.rs `fields_off`)
};
// A jump's working space.
struct JumpSpace{float2 u[3][MAXN],f[MAXN];float weights[MOST_CLASSES],shares[MOST_BEAMS];int beams[MOST_BEAMS];};
struct Warp{
    Point pt;
    float2 psi[MAXM],start[MAXM];       // ψ, ψ at a step's start
    double2 readout[MAXN];              // the readout's Y ψ
    float2 twin[MAXN];                  // the held twin's ψ
    // vec: the exact step's vectors (`half_of`); dressing: the exact responses' shifts ΔE_a^d (`exact_responses`).
    union{float2 m[3][MAXN*MAXN];float2 gradient_a[MOST_BEAMS][3];JumpSpace jump;float4 vec[MAXM/2];float dressing[MOST_CARRIERS*NG];};
#ifdef PROFILE
    long long profile[SPANS];
#endif
};

// The trajectory's scalars (uniform over the warp).
struct Traj{double q[3],p[3],time,logn,threshold,log_threshold;float doppler;int cluster,jumps,death;u32 index;float beats;};
// The held twin's (control.rs): it does not move.
struct Twin{double logn,threshold,log_threshold;int cluster,jumps;};

// A lane's entries of an n × n matrix: e = lane + 32 j, its row and column.
struct Entries{
    int n,nn,row[SLOTS],col[SLOTS];
    __device__ Entries(int size,int lane){
        n=size;nn=n*n;
        #pragma unroll
        for(int j=0;j<SLOTS;j++){const int e=lane+32*j;row[j]=e/n;col[j]=e-row[j]*n;}
    }
    __device__ bool has(int j,int lane)const{return lane+32*j<nn;}
};

__device__ __forceinline__ void lab(const Program& P,const double v[3],double out[3]){
    #pragma unroll
    for(int a=0;a<3;a++)out[a]=P.axes[0][a]*v[0]+P.axes[1][a]*v[1]+P.axes[2][a]*v[2];
}
// The same in single precision: where the point is evaluated and the windows' values summed.
__device__ __forceinline__ void lab32(const Program& P,const double v[3],float out[3]){
    const float x=(float)v[0],y=(float)v[1],z=(float)v[2];
    #pragma unroll
    for(int a=0;a<3;a++)out[a]=P.axes32[0][a]*x+P.axes32[1][a]*y+P.axes32[2][a]*z;
}

// ---------------------------------------------------------------------------------------------
// The point.

// a(r) and ∇a(r) (model.rs `Geometry::field`): the plane wave e^{ikζ}, or the paraxial Gaussian
// beam focused at the origin e^{ikζ} exp(−ρ²/(w² q))/q, q = 1 + iζ/z_R.
__device__ void field(const Beam& b,const float r[3],float2& a,float2 g[3]){
    const float zeta=r[0]*b.direction[0]+r[1]*b.direction[1]+r[2]*b.direction[2];
    float sn,cs;sincosf(b.k*zeta,&sn,&cs);const float2 wave=make_float2(cs,sn);
    if(b.inv_w2==0.f){
        a=wave;const float2 ika=make_float2(-b.k*sn,b.k*cs);
        #pragma unroll
        for(int i=0;i<3;i++)g[i]=b.direction[i]*ika;
        return;
    }
    const float2 inv=cdiv(make_float2(1.f,0.f),make_float2(1.f,zeta*b.inv_rayleigh));
    const float rho2=fmaxf(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]-zeta*zeta,0.f);
    a=wave*cexp((-rho2*b.inv_w2)*inv)*inv;
    const float2 along=make_float2(0.f,b.k)+b.inv_rayleigh*(make_float2(0.f,rho2*b.inv_w2)*inv*inv-make_float2(-inv.y,inv.x));
    const float2 across=(-2.f*b.inv_w2)*inv;
    #pragma unroll
    for(int i=0;i<3;i++)g[i]=a*(b.direction[i]*along+(r[i]-zeta*b.direction[i])*across);
}

// Cubic Lagrange weights on four consecutive nodes of a uniform grid of `nodes` points on [0, 1]
// at s, and their derivatives in s (model.rs `cubic`).
__device__ void cubic(int nodes,float s,int& first,float w[4],float d[4]){
    const float h=(float)(nodes-1);const float x=fminf(fmaxf(s,0.f),1.f)*h;
    first=min(max((int)floorf(x)-1,0),nodes-4);const float t=x-(float)first;
    w[0]=-(t-1.f)*(t-2.f)*(t-3.f)/6.f;w[1]=t*(t-2.f)*(t-3.f)/2.f;w[2]=-t*(t-1.f)*(t-3.f)/2.f;w[3]=t*(t-1.f)*(t-2.f)/6.f;
    d[0]=-((t-2.f)*(t-3.f)+(t-1.f)*(t-3.f)+(t-1.f)*(t-2.f))/6.f*h;d[1]=((t-2.f)*(t-3.f)+t*(t-3.f)+t*(t-2.f))/2.f*h;
    d[2]=-((t-1.f)*(t-3.f)+t*(t-3.f)+t*(t-1.f))/2.f*h;d[3]=((t-1.f)*(t-2.f)+t*(t-2.f)+t*(t-1.f))/6.f*h;
}

// The point of the trajectory (model.rs `Model::point`): every beam's amplitude and kinetic shift,
// the focused units' intensities s = |F|²/N (F = Σ_l a_l γ_l), their gradients 2 Re(F* ∇F)/N and
// stencils (and with the trap's light off), the carriers' fields (and with the trap's light off).
__device__ void point(const Program& P,Warp& W,const Traj& t,int lane){
    float r[3],kappa[3];lab32(P,t.q,r);lab32(P,t.p,kappa);
    Point& pt=W.pt;
    __syncwarp();
    if(lane<3){pt.r[lane]=r[lane];pt.kappa[lane]=kappa[lane];}
    if(lane<P.nbeams){
        const Beam& b=P.beams[lane];float2 a,g[3];field(b,r,a,g);
        pt.a[lane]=a;pt.shift[lane]=P.doppler?P.hm*(b.wave[0]*kappa[0]+b.wave[1]*kappa[1]+b.wave[2]*kappa[2])+b.recoil:0.f;
        #pragma unroll
        for(int i=0;i<3;i++)W.gradient_a[lane][i]=g[i];
    }
    __syncwarp();
    if(lane<P.focused){
        const Unit U=P.units[P.focused_unit[lane]];float2 F=zero2(),rest=zero2(),G[3]={zero2(),zero2(),zero2()};
        for(int j=0;j<U.count;j++){
            const int l=P.unit_beams[U.first+j];const float2 w=P.unit_weights[U.first+j],x=pt.a[l]*w;F+=x;if(!P.beams[l].trap)rest+=x;
            #pragma unroll
            for(int i=0;i<3;i++)G[i]+=W.gradient_a[l][i]*w;
        }
        const float s=fminf(norm2(F)/U.norm,1.f),off=fminf(norm2(rest)/U.norm,1.f);pt.s[lane]=s;
        #pragma unroll
        for(int i=0;i<3;i++)pt.grad_s[lane][i]=2.f*(F.x*G[i].x+F.y*G[i].y)/U.norm;
        cubic(P.nodes,s,pt.first[lane],pt.weight[lane],pt.slope[lane]);
        float unused[4];cubic(P.nodes,off,pt.far_first[lane],pt.far_weight[lane],unused);
    }
    if(lane<P.fields){
        const int c=P.field_carrier[lane],k=P.field_rank[lane];const Carrier C=P.carriers[c];
        float2 e=zero2(),o=zero2(),d=zero2(),g[3]={zero2(),zero2(),zero2()};
        for(int j=0;j<C.beams;j++){
            const int l=P.beam_list[C.first+j];const float2 x=make_float2(P.beams[l].coefficients[k][0],P.beams[l].coefficients[k][1]);
            const float2 ax=pt.a[l]*x;e+=ax;d+=pt.shift[l]*ax;if(!P.beams[l].trap)o+=ax;
            #pragma unroll
            for(int i=0;i<3;i++)g[i]+=W.gradient_a[l][i]*x;
        }
        Fields& f=pt.fields[c];f.e[k]=e;f.doppler[k]=d;pt.off[c][k]=o;
        #pragma unroll
        for(int i=0;i<3;i++)f.grad[k][i]=g[i];
    }
    __syncwarp();
}

// The first order's expansion parameter at κ (model.rs `doppler_ratio`): the largest (w_l/σ_c)²
// over the near beams.
__device__ float doppler_ratio(const Program& P,const Traj& t,int lane){
    float k[3];lab32(P,t.p,k);float v=0.f;
    if(lane<P.nbeams){
        const Beam& b=P.beams[lane];
        if(b.inv_pole>0.f){const float w=P.doppler?P.hm*(b.wave[0]*k[0]+b.wave[1]*k[1]+b.wave[2]*k[2])+b.recoil:0.f;const float x=w*b.inv_pole;v=x*x;}
    }
    return warp_max(v);
}

// ---------------------------------------------------------------------------------------------
// The tables at the point.

// The stencil's nodes come in rows of four along the first focused beam's axis (stride 1), one row
// per node of the others': a row's four loads are issued together, the rows in the node order of
// model.rs `Stencil::node` (the first axis fastest). Of a row: its first node and the other axes'
// weights, with their derivatives' (`dw`, per axis, the row's factor of the first axis's own).
struct Row{int index;float w,dw[MOST_FOCUSED];};
__device__ __forceinline__ Row stencil_row(const Program& P,const Point& pt,bool far,int o){
    Row r;r.index=far?pt.far_first[0]:pt.first[0];r.w=1.f;
    #pragma unroll
    for(int g=0;g<MOST_FOCUSED;g++)r.dw[g]=1.f;
    #pragma unroll
    for(int f=1;f<MOST_FOCUSED;f++)if(f<P.focused){
        const int m=(o>>(2*(f-1)))&3;const float wf=far?pt.far_weight[f][m]:pt.weight[f][m];
        r.index+=((far?pt.far_first[f]:pt.first[f])+m)*P.stride[f];r.w*=wf;
        #pragma unroll
        for(int g=1;g<MOST_FOCUSED;g++)r.dw[g]*=g==f?pt.slope[f][m]:wf;
    }
    return r;
}

// A slice of `count` entries of a cluster's tables at the stencil (model.rs `gather`), with the trap
// beam off when `far`: the lane's entries; with SLOPES their derivatives in every s_f too.
template<bool SLOPES>
__device__ void gather(const Program& P,const Warp& W,bool far,const float2* base,int size,int count,int lane,float2 (&value)[SLOTS],float2 (&slope)[MOST_FOCUSED][SLOTS]){
    const Point& pt=W.pt;const int D=P.focused;
    #pragma unroll
    for(int j=0;j<SLOTS;j++){
        value[j]=zero2();
        #pragma unroll
        for(int g=0;g<MOST_FOCUSED;g++)slope[g][j]=zero2();
    }
    const float* w0=far?pt.far_weight[0]:pt.weight[0];const float* s0=pt.slope[0];
    for(int o=0;o<(1<<(2*(D-1)));o++){
        const Row r=stencil_row(P,pt,far,o);const float2* x=base+(size_t)r.index*size;
        #pragma unroll
        for(int j=0;j<SLOTS;j++){
            const int e=lane+32*j;
            if(e<count){
                float2 v[4];
                #pragma unroll
                for(int m=0;m<4;m++)v[m]=__ldg(x+(size_t)m*size+e);
                #pragma unroll
                for(int m=0;m<4;m++){
                    value[j]+=(r.w*w0[m])*v[m];
                    if(SLOPES){
                        slope[0][j]+=(r.w*s0[m])*v[m];
                        #pragma unroll
                        for(int g=1;g<MOST_FOCUSED;g++)if(g<D)slope[g][j]+=(r.dw[g]*w0[m])*v[m];
                    }
                }
            }
        }
    }
}

// One entry of a cluster's tables at the stencil, or its derivative in s_f (model.rs `interpolate`).
__device__ float2 entry(const Program& P,const Warp& W,const float2* x,int size,int derivative){
    const Point& pt=W.pt;float2 v=zero2();const float* w0=derivative==0?pt.slope[0]:pt.weight[0];
    for(int o=0;o<(1<<(2*(P.focused-1)));o++){
        const Row r=stencil_row(P,pt,false,o);const float factor=derivative>0?r.dw[derivative]:r.w;const float2* at=x+(size_t)r.index*size;
        float2 u[4];
        #pragma unroll
        for(int m=0;m<4;m++)u[m]=__ldg(at+(size_t)m*size);
        #pragma unroll
        for(int m=0;m<4;m++)v+=(factor*w0[m])*u[m];
    }
    return v;
}
// One entry of a cluster's tables at the stencil with the trap beam off (`gather`'s `far`).
__device__ float2 far_entry(const Program& P,const Warp& W,const float2* x,int size){
    const Point& pt=W.pt;float2 v=zero2();const float* w0=pt.far_weight[0];
    for(int o=0;o<(1<<(2*(P.focused-1)));o++){
        const Row r=stencil_row(P,pt,true,o);const float2* at=x+(size_t)r.index*size;
        float2 u[4];
        #pragma unroll
        for(int m=0;m<4;m++)u[m]=__ldg(at+(size_t)m*size);
        #pragma unroll
        for(int m=0;m<4;m++)v+=(r.w*w0[m])*u[m];
    }
    return v;
}
// One entry of a cluster's tables at the stencil with its derivatives in every s_f (`gather`'s of one entry).
__device__ float2 entry_slopes(const Program& P,const Warp& W,const float2* x,int size,float2 (&slope)[MOST_FOCUSED]){
    const Point& pt=W.pt;float2 v=zero2();const float* w0=pt.weight[0];const float* s0=pt.slope[0];
    #pragma unroll
    for(int g=0;g<MOST_FOCUSED;g++)slope[g]=zero2();
    for(int o=0;o<(1<<(2*(P.focused-1)));o++){
        const Row r=stencil_row(P,pt,false,o);const float2* at=x+(size_t)r.index*size;
        float2 u[4];
        #pragma unroll
        for(int m=0;m<4;m++)u[m]=__ldg(at+(size_t)m*size);
        #pragma unroll
        for(int m=0;m<4;m++){
            v+=(r.w*w0[m])*u[m];slope[0]+=(r.w*s0[m])*u[m];
            #pragma unroll
            for(int g=1;g<MOST_FOCUSED;g++)if(g<P.focused)slope[g]+=(r.dw[g]*w0[m])*u[m];
        }
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// The generator.

__device__ __forceinline__ bool zero(float2 z){return z.x==0.f&&z.y==0.f;}

// Σ over a cluster's carriers c of Σ_k1k2 E^c*_k1 E^c_k2 X^c_k1k2 at the lane's entries, X^c the r_c²
// matrices n × n from slices[c] (−1: none) of its fourth-order tables, at the point: a quadratic form
// of the fields (model.rs `fourth_quadratic`).
__device__ void fourth_quadratic(const Program& P,const Warp& W,const Cluster& C,const Fourth& F,const int (&slices)[MOST_CARRIERS],float scale,int lane,float2 (&out)[SLOTS]){
    const float2* tables=P.tables+C.table;const int nn=C.n*C.n;float2 buf[SLOTS],unused[MOST_FOCUSED][SLOTS];
    for(int ci=0;ci<F.on;ci++){
        const int c=F.carriers[ci],rc=P.carriers[c].rank,at=slices[c];if(at<0)continue;const Fields& fc=W.pt.fields[c];
        for(int k1=0;k1<rc;k1++)for(int k2=0;k2<rc;k2++){
            const float2 m=scale*(conj(fc.e[k1])*fc.e[k2]);if(zero(m))continue;
            gather<false>(P,W,false,tables+at+(k1*rc+k2)*nn,C.size,nn,lane,buf,unused);
            #pragma unroll
            for(int j=0;j<SLOTS;j++)out[j]+=m*buf[j];
        }
    }
}
// The lane's entries of an n × n matrix in the cluster's frames at t: entry (a, b) times e^{i(θ_a − θ_b)t}.
__device__ void in_frames(const Program& P,const Cluster& C,const Entries& E,double t,int lane,float2 (&x)[SLOTS]){
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)&&E.row[j]!=E.col[j])x[j]=x[j]*cis((P.frames[C.frames+E.row[j]]-P.frames[C.frames+E.col[j]])*t);
}
// A cluster's fourth-order part of H_eff at the point and t at the lane's entries (model.rs
// `fourth_hamiltonian`): Σ_(c,d) Σ E^c*E^c E^d*E^d X over its quartic tables and Σ_c Σ E^c*E^c ½[R†R, H_g],
// in the cluster's frames; zero for a cluster folded at second order.
__device__ void fourth_hamiltonian(const Program& P,const Warp& W,const Cluster& C,const Entries& E,double t,int lane,float2 (&h)[SLOTS]){
    #pragma unroll
    for(int j=0;j<SLOTS;j++)h[j]=zero2();
    if(C.fourth<0)return;
    const Fourth& F=P.fourths[C.fourth];const float2* tables=P.tables+C.table;const int nn=E.nn;const Point& pt=W.pt;
    float2 buf[SLOTS],unused[MOST_FOCUSED][SLOTS];
    for(int ci=0;ci<F.on;ci++){
        const int c=F.carriers[ci],rc=P.carriers[c].rank;const Fields& fc=pt.fields[c];
        for(int di=0;di<F.on;di++){
            const int d=F.carriers[di],rd=P.carriers[d].rank,at=F.quartic[c][d];if(at<0)continue;const Fields& fd=pt.fields[d];
            for(int k1=0;k1<rc;k1++)for(int k2=0;k2<rc;k2++){
                const float2 x=conj(fc.e[k1])*fc.e[k2];if(zero(x))continue;
                for(int k3=0;k3<rd;k3++)for(int k4=0;k4<rd;k4++){
                    const float2 m=x*(conj(fd.e[k3])*fd.e[k4]);if(zero(m))continue;
                    gather<false>(P,W,false,tables+at+(((k1*rc+k2)*rd+k3)*rd+k4)*nn,C.size,nn,lane,buf,unused);
                    #pragma unroll
                    for(int j=0;j<SLOTS;j++)h[j]+=m*buf[j];
                }
            }
        }
    }
    fourth_quadratic(P,W,C,F,F.quadratic,1.f,lane,h);
    in_frames(P,C,E,t,lane,h);
}

// A cluster's generator at the point and time t: H_eff less its scalar part into W.m[0] (row-major,
// model.rs `Local::hamiltonian`), and in the lane's entries h0 and the gradient of the Hermitian
// part per axis (`Local::gradient`).
struct Local{float2 h0[SLOTS],grad[3][SLOTS];};
__device__ void local(const Program& P,Warp& W,const Cluster& C,const Entries& E,double t,int lane,Local& L){
    const int nn=E.nn;const float2* tables=P.tables+C.table;const Point& pt=W.pt;const int D=P.focused;
    float2 m[SLOTS],ma[SLOTS],g[SLOTS],ga[SLOTS],buf[SLOTS],slope[MOST_FOCUSED][SLOTS];
    #pragma unroll
    for(int j=0;j<SLOTS;j++){
        m[j]=ma[j]=g[j]=ga[j]=zero2();
        #pragma unroll
        for(int a=0;a<3;a++)L.grad[a][j]=zero2();
    }
    for(int gi=0;gi<C.ngroups;gi++){
        const Group G=P.groups[C.groups+gi];if(G.count!=nn)continue;
        const Fields& fa=pt.fields[G.absorbed];const Fields& fe=pt.fields[G.emitted];
        const float2 phase=residual(G.residual,t);
        for(int pi=0;pi<G.npairs;pi++){
            const int2 pair=P.pairs[G.pairs+pi];const int k1=pair.x,k=pair.y;
            // A Hermitian group's pair (k1, k) stands for (k, k1) too: (ee S)† into the adjoint half;
            // the force reads Re⟨ψ|·|ψ⟩ only, so its matrices take the pair twice.
            const bool twice=G.hermitian&&k1<k;const float both=twice?2.f:1.f;
            const float2 ee=phase*conj(fe.e[k1])*fa.e[k];
            gather<true>(P,W,false,tables+G.s+pi*nn,C.size,nn,lane,buf,slope);
            #pragma unroll
            for(int j=0;j<SLOTS;j++){m[j]+=ee*buf[j];if(twice)ma[j]+=ee*buf[j];}
            #pragma unroll
            for(int a=0;a<3;a++){
                const float2 c=(-both)*(phase*(conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a]));
                #pragma unroll
                for(int j=0;j<SLOTS;j++)L.grad[a][j]+=c*buf[j];
                #pragma unroll
                for(int f=0;f<MOST_FOCUSED;f++)if(f<D){
                    const float2 cf=(-both*pt.grad_s[f][a])*ee;
                    #pragma unroll
                    for(int j=0;j<SLOTS;j++)L.grad[a][j]+=cf*slope[f][j];
                }
            }
            gather<false>(P,W,false,tables+G.q+pi*nn,C.size,nn,lane,buf,slope);
            #pragma unroll
            for(int j=0;j<SLOTS;j++){g[j]+=ee*buf[j];if(twice)ga[j]+=ee*buf[j];}
        }
    }
    // The sparse groups (fewer entries than places, model.rs `scatter`): each place's own entries, an
    // entry and, from a pair standing for two, its adjoint at the mirror place; their beats' phases
    // once (in W.m[2], free until the adjoint halves below). A cluster of full groups alone has none.
    if(C.nbeats>0){
        float2* phases=W.m[2];
        __syncwarp();
        if(lane<C.nbeats)phases[lane]=residual(P.residuals[C.beats+lane],t);
        __syncwarp();
        #pragma unroll
        for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){
            const int e=lane+32*j,end=P.item_first[C.positions+e+1];
            #pragma unroll 2
            for(int i=P.item_first[C.positions+e];i<end;i++){
                const int4 it=__ldg(P.items+i);
                float2 ds[MOST_FOCUSED];const float2 s=entry_slopes(P,W,tables+it.x,C.size,ds);const float2 q=entry(P,W,tables+it.y,C.size,-1);
                const int k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
                const float2 phase=phases[it.z>>12];const float2 ee=phase*conj(fe.e[k1])*fa.e[k];
                if((it.z>>11)&1){m[j]+=conj(ee*s);g[j]+=conj(ee*q);continue;}
                m[j]+=ee*s;g[j]+=ee*q;
                const float both=(it.z>>10)&1?2.f:1.f;
                #pragma unroll
                for(int a=0;a<3;a++){
                    L.grad[a][j]+=((-both)*(phase*(conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a])))*s;
                    #pragma unroll
                    for(int f=0;f<MOST_FOCUSED;f++)if(f<D)L.grad[a][j]+=((-both*pt.grad_s[f][a])*ee)*ds[f];
                }
            }
        }
    }
    gather<true>(P,W,false,tables+C.h0,C.size,nn,lane,L.h0,slope);
    #pragma unroll
    for(int a=0;a<3;a++){
        #pragma unroll
        for(int f=0;f<MOST_FOCUSED;f++)if(f<D){
            const float c=pt.grad_s[f][a];
            #pragma unroll
            for(int j=0;j<SLOTS;j++)L.grad[a][j]+=c*slope[f][j];
        }
    }
    // The adjoint halves through shared memory; the scalar part; H_eff = h0 − M − (i/2)Γ less it.
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){const int e=lane+32*j;W.m[1][e]=ma[j];W.m[2][e]=ga[j];}
    __syncwarp();
    float diagonal=0.f;
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){
        const int mirror=E.col[j]*E.n+E.row[j];m[j]+=conj(W.m[1][mirror]);g[j]+=conj(W.m[2][mirror]);
        if(E.row[j]==E.col[j])diagonal+=L.h0[j].x-m[j].x;
    }
    const float scalar=warp_sum(diagonal)/(float)E.n;
    // The fourth order's part (model.rs `fourth_hamiltonian`), in the cluster's frames; the force keeps
    // the second order's.
    float2 h4[SLOTS];
    fourth_hamiltonian(P,W,C,E,t,lane,h4);
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){
        float2 h=make_float2(L.h0[j].x-m[j].x+0.5f*g[j].y,L.h0[j].y-m[j].y-0.5f*g[j].x);
        if(E.row[j]==E.col[j])h.x-=scalar;
        W.m[0][lane+32*j]=h+h4[j];
    }
    __syncwarp();
}

// out = A B for n × n matrices in shared memory: the lane's entries.
__device__ __forceinline__ void product(const float2* A,const float2* B,const Entries& E,int lane,float2 (&out)[SLOTS]){
    #pragma unroll
    for(int j=0;j<SLOTS;j++){
        float2 s=zero2();
        if(E.has(j,lane)){
            const float2* x=A+E.row[j]*E.n;const float2* y=B+E.col[j];
            #pragma unroll
            for(int c=0;c<MAXN;c++)if(c<E.n)s+=x[c]*y[c*E.n];
        }
        out[j]=s;
    }
}

// U = exp(−iHτ) for the generator in W.m[0] (engine.rs `propagate`, on the matrix): sub-steps of
// ‖H‖τ ≤ ½, ‖H‖ the largest column sum of |re| + |im|, each the Taylor polynomial P of
// X = −iHτ/parts through the term whose bound falls below its share of the truncation; U = P^parts,
// the propagation of every vector by `parts` sub-steps of P at once. U into W.m[2].
__device__ void exponential(const Program& P,Warp& W,const Entries& E,double tau,int lane){
    float column=0.f;
    if(lane<E.n)for(int a=0;a<E.n;a++){const float2 h=W.m[0][a*E.n+lane];column+=fabsf(h.x)+fabsf(h.y);}
    const float x=warp_max(column)*fabsf((float)tau);const int parts=max((int)ceilf(2.f*x),1);const float step=(float)tau/(float)parts;
    const float y=x/(float)parts,share=P.truncation/(float)parts;int terms=1;float size=y;while(size>share&&terms<40){terms++;size*=y/(float)terms;}
    // X = −i step H into m[0]; the term (m[1]) X and the sum (m[2]) I + X.
    float2 v[SLOTS];
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){const float2 h=W.m[0][lane+32*j];v[j]=make_float2(h.y*step,-h.x*step);}
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){const int e=lane+32*j;W.m[0][e]=v[j];W.m[1][e]=v[j];W.m[2][e]=v[j]+make_float2(E.row[j]==E.col[j]?1.f:0.f,0.f);}
    __syncwarp();
    for(int k=2;k<=terms;k++){
        product(W.m[1],W.m[0],E,lane,v);
        const float inverse=1.f/(float)k;
        __syncwarp();
        #pragma unroll
        for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){const int e=lane+32*j;v[j]=inverse*v[j];W.m[1][e]=v[j];W.m[2][e]+=v[j];}
        __syncwarp();
    }
    if(parts>1){
        #pragma unroll
        for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){const int e=lane+32*j;W.m[0][e]=W.m[2][e];}
        __syncwarp();
        for(int p=1;p<parts;p++){
            product(W.m[2],W.m[0],E,lane,v);
            __syncwarp();
            #pragma unroll
            for(int j=0;j<SLOTS;j++)if(E.has(j,lane))W.m[2][lane+32*j]=v[j];
            __syncwarp();
        }
    }
}

// ψ ← U ψ (n × n, row-major), renormalized; returns ‖Uψ‖² relative to the ‖ψ‖² = 1 it had.
__device__ float apply(const float2* U,float2* psi,int n,int lane){
    float2 v=zero2();
    if(lane<n){
        const float2* u=U+lane*n;
        #pragma unroll
        for(int b=0;b<MAXN;b++)if(b<n)v+=u[b]*psi[b];
    }
    const float s=warp_sum(lane<n?norm2(v):0.f);
    __syncwarp();if(lane<n)psi[lane]=(1.f/sqrtf(s))*v;__syncwarp();
    return s;
}

// Re⟨ψ|X|ψ⟩/⟨ψ|ψ⟩ of the lane's entries of X.
__device__ float quadratic(const Warp& W,const Entries& E,const float2 (&x)[SLOTS],int lane){
    float s=0.f;
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){const float2 pa=W.psi[E.row[j]],z=x[j]*W.psi[E.col[j]];s+=pa.x*z.x+pa.y*z.y;}
    float norm=0.f;for(int b=0;b<E.n;b++)norm+=norm2(W.psi[b]);
    return warp_sum(s)/norm;
}

// ---------------------------------------------------------------------------------------------
// The motion.

// The harmonic part's weight at a step's start (engine.rs `Engine::weight`): one within the trap's
// region, U_h ≤ D, (2u − 1)/u² beyond it, u = U_h/D; one without a region.
__device__ double weight(const Program& P,const double* q){
    if(!(P.region>0.))return 1.;
    double harmonic=0.;
    #pragma unroll
    for(int i=0;i<3;i++)harmonic+=P.omega[i]*P.omega[i]*q[i]*q[i];
    const double u=0.5*harmonic/P.hm64/P.region;
    return u<=1.?1.:(2.*u-1.)/(u*u);
}

// The harmonic flow for τ along each principal axis, its part weighted by `weight` (engine.rs
// `flow`): (x, κ) turned by ω√w τ in phase space; free flight when ω is 0.
__device__ void flow(const Program& P,Traj& t,double tau,double weight){
    const bool half=tau==0.5*P.dt&&weight==1.;const double scale=weight==1.?1.:sqrt(weight);
    #pragma unroll
    for(int i=0;i<3;i++){
        const double w=P.omega[i]*scale;double c,sh,sw;
        if(half){c=P.half_flow[i][0];sh=P.half_flow[i][1];sw=P.half_flow[i][2];}
        else if(w==0.){c=1.;sh=P.hm64*tau;sw=0.;}
        else{double s;sincos(w*tau,&s,&c);sh=P.hm64/w*s;sw=w/P.hm64*s;}
        const double x=t.q[i],k=t.p[i];t.q[i]=x*c+k*sh;t.p[i]=k*c-x*sw;
    }
}

// The step's middle state the loss is judged at (engine.rs `Engine::step`): q and κ -- with half the
// kick -- on the trap's principal axes, and the lane's entry of ψ after the first half step.
struct Middle{double q[3],p[3];float2 psi;};

// One step's flow, kick and internal evolution of length dt without jumps (engine.rs `advance`):
// ln‖ψ‖² at the middle (`lh`) and the end (t.logn), the midpoint's point in W.pt and h0 there in L, the
// middle state in M.
__device__ void advance(const Program& P,Warp& W,Traj& t,double dt,int lane,double& lh,Local& L,Middle& M){
    COUNT(COUNT_ADVANCES);SPAN_BEGIN(start);
    const double w=weight(P,t.q);
    flow(P,t,0.5*dt,w);
    point(P,W,t,lane);
    SPAN_END(SPAN_POINT,start);SPAN_BEGIN(generator);
    const Cluster C=P.clusters[t.cluster];const Entries E(C.n,lane);
    local(P,W,C,E,t.time+0.5*dt,lane,L);
    SPAN_END(SPAN_GENERATOR,generator);SPAN_BEGIN(propagator);
    exponential(P,W,E,0.5*dt,lane);
    SPAN_END(SPAN_EXPONENTIAL,propagator);SPAN_BEGIN(evolve);
    const float nh=apply(W.m[2],W.psi,C.n,lane);lh=t.logn+(double)logf(nh);
    if(lane<MAXN)M.psi=W.psi[lane];
    // The Ehrenfest force −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ at the middle, less the weighted harmonic part the flow takes.
    float s[3]={0.f,0.f,0.f};
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){
        const float2 pa=W.psi[E.row[j]],pb=W.psi[E.col[j]];
        #pragma unroll
        for(int a=0;a<3;a++){const float2 z=L.grad[a][j]*pb;s[a]+=pa.x*z.x+pa.y*z.y;}
    }
    float norm=0.f;for(int b=0;b<C.n;b++)norm+=norm2(W.psi[b]);
    float force[3];
    #pragma unroll
    for(int a=0;a<3;a++)force[a]=-warp_sum(s[a])/norm;
    const float tau=(float)dt;
    #pragma unroll
    for(int i=0;i<3;i++){
        const double kick=(double)(tau*(P.axes32[i][0]*force[0]+P.axes32[i][1]*force[1]+P.axes32[i][2]*force[2]+(float)w*P.stiffness32[i]*(float)t.q[i]));
        M.q[i]=t.q[i];M.p[i]=t.p[i]+0.5*kick;t.p[i]+=kick;
    }
    const float n1=apply(W.m[2],W.psi,C.n,lane);t.logn=lh+(double)logf(n1);
    flow(P,t,0.5*dt,w);
    t.time+=dt;
    SPAN_END(SPAN_EVOLVE,evolve);
}

// ---------------------------------------------------------------------------------------------
// The jump.

__device__ __forceinline__ int exact_at(int near,int col,int a,int to){return ((near*MAXN+col)*3+a)*NG+to;}

// The LU factors of the NE × NE matrix whose row i lane i holds, by Gaussian elimination with partial
// pivoting as linear.rs `Lu` makes it (each column's pivot the remaining row of the largest |a|², the
// last in the rows' order on ties), the rows staying in their lanes' registers and only their places in
// that order (`at`) swapped: a row keeps U's entries from its place on, and before it, in each step's
// column, that step's multiplier. The steps a loop, a column's entry selected (the code stays small for
// the driver's JIT). False when a pivot vanishes.
__device__ __forceinline__ bool factor(float2 (&a)[NE],int lane,int& at){
    at=lane;
    for(int k=0;k<NE;k++){
        float2 ak=zero2();
        #pragma unroll
        for(int j=0;j<NE;j++)if(j==k)ak=a[j];
        // The pivot: the largest |a_ik|² of the rows at k or later, the last of them on ties.
        float best=lane<NE&&at>=k?norm2(ak):-1.f;int place=at,who=lane;
        #pragma unroll
        for(int o=16;o>0;o>>=1){
            const float ob=__shfl_xor_sync(FULL,best,o);const int op=__shfl_xor_sync(FULL,place,o),ow=__shfl_xor_sync(FULL,who,o);
            if(ob>best||(ob==best&&op>place)){best=ob;place=op;who=ow;}
        }
        if(!(best>0.f))return false;
        if(at==k)at=place;else if(lane==who)at=k;
        // The rows after it less their multiple of the pivot's row, the multiplier kept in its column.
        const float2 pivot=make_float2(__shfl_sync(FULL,ak.x,who),__shfl_sync(FULL,ak.y,who));
        const float2 m=at>k?cdiv(ak,pivot):zero2();
        #pragma unroll
        for(int j=0;j<NE;j++){
            if(j>k){const float2 v=make_float2(__shfl_sync(FULL,a[j].x,who),__shfl_sync(FULL,a[j].y,who));a[j]-=m*v;}
            else if(j==k&&at>k)a[j]=m;
        }
    }
    return true;
}
// x = A⁻¹ b from `factor`'s rows, the lane's row's entry of b in `b`: forward through the steps'
// multipliers in their order, then back from the last place (x_k from its row, the rows before it less
// U_ik x_k); each x_k, uniform over the warp, to `take` as it is found.
template<class Take>
__device__ __forceinline__ void substitute(const float2 (&a)[NE],int at,float2 b,int lane,Take take){
    for(int k=0;k<NE;k++){
        float2 ak=zero2();
        #pragma unroll
        for(int j=0;j<NE;j++)if(j==k)ak=a[j];
        const int who=__ffs(__ballot_sync(FULL,at==k))-1;
        const float2 bk=make_float2(__shfl_sync(FULL,b.x,who),__shfl_sync(FULL,b.y,who));
        if(at>k)b-=ak*bk;
    }
    for(int k=NE-1;k>=0;k--){
        float2 ak=zero2();
        #pragma unroll
        for(int j=0;j<NE;j++)if(j==k)ak=a[j];
        const int who=__ffs(__ballot_sync(FULL,at==k))-1;
        const float2 mine=at==k?cdiv(b,ak):zero2();
        const float2 x=make_float2(__shfl_sync(FULL,mine.x,who),__shfl_sync(FULL,mine.y,who));
        take(k,x);
        if(at<k)b-=ak*x;
    }
}

// The point's H diagonal in double precision (`Program::diagonal`): ground state b at x = b, excited
// state i at x = NG + i.
__device__ __forceinline__ double diagonal_at(const Program& P,const Point& pt,int x){
    double d=P.diagonal[x];for(int f=0;f<P.focused;f++)d+=(double)pt.s[f]*P.diagonal[(1+f)*(NG+NE)+x];return d;
}
// K's entry (i, j) at the point in single precision (`Program::excited`).
__device__ __forceinline__ float2 excited_at(const Program& P,const Point& pt,int i,int j){
    const int e=i*NE+j;float2 k=P.excited[e];for(int f=0;f<P.focused;f++)k+=pt.s[f]*P.excited[(1+f)*NE*NE+e];return k;
}
// Whether K's decoupling from the explicit states p (bits; ground.rs `Ground::decoupled`) keeps entry (i, j).
__device__ __forceinline__ bool kept_entry(u32 p,int i,int j){return i==j||!(((p>>i)|(p>>j))&1u);}
// The lane's row of K − z at the point, decoupled from the explicit states p, its diagonal's real part
// H_ii − z in double precision.
__device__ __forceinline__ void resolvent_row(const Program& P,const Point& pt,u32 p,double z,int lane,float2 (&a)[NE]){
    #pragma unroll
    for(int j=0;j<NE;j++){
        float2 k=zero2();
        if(lane<NE&&kept_entry(p,lane,j)){k=excited_at(P,pt,lane,j);if(j==lane)k.x=(float)(diagonal_at(P,pt,NG+lane)-z);}
        a[j]=k;
    }
}
// A near unit's s N at the point: a focused unit at its intensity there, another at its focus's.
__device__ __forceinline__ float unit_weight(const Point& pt,const NearUnit& U){return U.norm*(U.focused>=0?pt.s[U.focused]:U.focus);}
// The self-energy Σ of channel (c, b) at z, added to the lane's row where the decoupling from p keeps
// its entries (ground.rs `Ground::dress_light`): over its far channels (d, a) and d's near units,
// s N W^f[:, a] W^f[:, a]†/(z − Ẽ_a − δ_d), Ẽ_a = E_a + Σ_d ΔE_a^d (the shifts in `shift`).
__device__ __forceinline__ void self_energy(const Program& P,const Point& pt,const float* shift,int c,int b,double z,u32 p,int lane,float2 (&a)[NE]){
    if(lane>=NE)return;
    for(int v=P.via_first[c*NG+b];v<P.via_first[c*NG+b+1];v++){
        const int2 via=P.vias[v];
        double e=diagonal_at(P,pt,via.y)+P.carriers[via.x].detuning;for(int d=0;d<P.ncarriers;d++)e+=(double)shift[d*NG+via.y];
        const float pole=1.f/(float)(z-e);
        for(int u=P.near_first[via.x];u<P.near_first[via.x+1];u++){
            const float2* w=P.unit_couplings+(u*NG+via.y)*NE;const float2 wi=(pole*unit_weight(pt,P.near_units[u]))*w[lane];
            #pragma unroll
            for(int j=0;j<NE;j++)if(kept_entry(p,lane,j))a[j]+=wi*conj(w[j]);
        }
    }
}
// The lane's entry of beam l's source from ground state b (ground.rs `Ground::source`): V_l^f[i, b] less,
// over the explicit states e of p with a kept drive from b, x_e[i] (V_l − V_l^f)[e, b] -- e's dressing
// x_e[i] = H_ie/(K_ee − K_ii) over its partners outside p (`Ground::dressing`).
__device__ __forceinline__ float2 source(const Program& P,const Point& pt,int l,int b,u32 p,int lane){
    if(lane>=NE)return zero2();
    float2 s=P.couplings[(l*NE+lane)*NG+b];
    if((p>>lane)&1u)return s;
    for(u32 m=p;m;m&=m-1){
        const int e=__ffs(m)-1;const float2 kept=P.kept[(l*NE+e)*NG+b];
        if(zero(kept)||!((P.partners[e]>>lane)&1u))continue;
        const float2 d=P.decay[lane*NE+e],h=excited_at(P,pt,lane,e)+make_float2(-0.5f*d.y,0.5f*d.x);
        const float2 gap=make_float2((float)(diagonal_at(P,pt,NG+e)-diagonal_at(P,pt,NG+lane)),(excited_at(P,pt,e,e)-excited_at(P,pt,lane,lane)).y);
        s-=cdiv(h,gap)*kept;
    }
    return s;
}

// The near light's shifts ΔE_a^d of a cluster's channels (`Cluster::shifts`, ground.rs `Ground::dress_light`)
// at the point into W.dressing (bit d·NG + a; zero where not taken): ΔE_a^d = −Σ_u s N Re(W^f[:, a]†
// (K − E_a − δ_d)⁻¹ W^f[:, a]) over d's near units, K decoupled from a's cluster's explicit states. False
// when a resolvent is singular.
__device__ __noinline__ bool light_shifts(const Program& P,Warp& W,const Cluster& C,int lane){
    const Point& pt=W.pt;float* shift=W.dressing;
    __syncwarp();shift[lane]=0.f;shift[lane+32]=0.f;__syncwarp();
    for(u64 m=C.shifts;m;m&=m-1){
        const int bit=__ffsll((long long)m)-1,d=bit/NG,a=bit-d*NG;
        float2 row[NE];resolvent_row(P,pt,P.explicit_of[a],diagonal_at(P,pt,a)+P.carriers[d].detuning,lane,row);
        int at;if(!factor(row,lane,at))return false;
        float v=0.f;
        for(int u=P.near_first[d];u<P.near_first[d+1];u++){
            const float2* w=P.unit_couplings+(u*NG+a)*NE;float2 dot=zero2();
            substitute(row,at,lane<NE?w[lane]:zero2(),lane,[&](int k,float2 x){dot+=conj(w[k])*x;});
            v-=unit_weight(pt,P.near_units[u])*dot.x;
        }
        if(lane==0)shift[bit]=v;
    }
    __syncwarp();
    return true;
}
// Channel (c, b)'s K + Σ − z at the point (`resolvent_row`, `self_energy`), decoupled from the explicit
// states p, into `rows` lane-major: entry j of row i at rows[32j + i].
__device__ __noinline__ void channel_row(const Program& P,const Warp& W,int c,int b,double z,u32 p,int lane,float2* rows){
    float2 a[NE];resolvent_row(P,W.pt,p,z,lane,a);self_energy(P,W.pt,W.dressing,c,b,z,p,lane,a);
    #pragma unroll
    for(int j=0;j<NE;j++)rows[j*32+lane]=a[j];
}

// The exact response's photon tables at the point (model.rs `Exact`, ground.rs `Ground::response`):
// X_l[a][to][col] = (C_a R_l(w_l))[to, b] for every near beam l and cluster column (b its ground state)
// at the point's intensities, dressed by the near light (ground.rs, # The near light's dressing):
// R_l[:, b] = (K + Σ − z + w_l)⁻¹ ŝ, channel (c, b)'s energy z = E_b + δ_c + Σ_{d≠c} ΔE_b^d (`light_shifts`)
// and self-energy Σ (`self_energy`), ŝ its source (`source`), K decoupled from the explicit states of b's
// cluster. A channel's K + Σ − z once (`channel_row`), each of its carrier's beams from it with w_l. Lane
// i holds row i of each system (`factor`) and lane 8a + to sums C_a R_l's entry. False when a resolvent is
// singular, as `Exact::new` fails (the jump keeps the first order).
__device__ __noinline__ bool exact_responses(const Program& P,Warp& W,const Cluster& C,int lane,float2* X,float2* rows){
    if(!light_shifts(P,W,C,lane))return false;
    const Point& pt=W.pt;const float* shift=W.dressing;const float2* y=lane<3*NG?P.emission+lane*NE:nullptr;
    for(int c=0;c<P.ncarriers;c++){
        const Carrier R=P.carriers[c];
        for(int col=0;col<C.n0;col++){
            const int g=C.states[col];const u32 p=P.explicit_of[g];
            u32 coupled=0u;
            for(int j=0;j<R.beams;j++){
                const int l=P.beam_list[R.first+j];if((P.beams[l].coupled>>g)&1u)coupled|=1u<<j;
                else if(lane<3*NG)X[exact_at(P.beams[l].near,col,0,0)+lane]=zero2();
            }
            if(!coupled)continue;
            double z=diagonal_at(P,pt,g)+R.detuning;for(int d=0;d<P.ncarriers;d++)if(d!=c)z+=(double)shift[d*NG+g];
            channel_row(P,W,c,g,z,p,lane,rows);
            for(;coupled;coupled&=coupled-1){
                const int l=P.beam_list[R.first+__ffs(coupled)-1];float2* out=X+exact_at(P.beams[l].near,col,0,0);
                float2 a[NE];const float w=pt.shift[l];
                #pragma unroll
                for(int j=0;j<NE;j++){a[j]=rows[j*32+lane];if(j==lane)a[j].x+=w;}
                const float2 s=source(P,pt,l,g,p,lane);
                int at;if(!factor(a,lane,at))return false;
                float2 response=zero2();substitute(a,at,s,lane,[&](int k,float2 x){if(y!=nullptr)response+=y[k]*x;});
                if(lane<3*NG)out[lane]=response;
            }
        }
    }
    return true;
}

// The fourth order's weights of a jump's photons (model.rs `fourth_photons`), the forms
// Σ E^d*_k3 E^e_k4 P^{de}_k3k4 on every ground state b′ × the cluster (ng × n): Σ_d P^{dd} at weights[0],
// R†R's Σ_c Σ E^c*E^c RR^c at weights[NG·n], P^{ec} (e ≠ c) at weights[(2 + e_i·on + c_i)·NG·n].
__device__ void fourth_weights(const Program& P,const Warp& W,const Cluster& C,int lane,float2* weights){
    const Fourth& F=P.fourths[C.fourth];const float2* tables=P.tables+C.table;const int n=C.n,size=NG*n;const Point& pt=W.pt;
    const int count=(2+F.on*F.on)*size;
    for(int i=lane;i<count;i+=32){
        const int which=i/size,at=i-which*size;float2 w=zero2();
        auto form=[&](int d,int e,int slice){
            const int rd=P.carriers[d].rank,re=P.carriers[e].rank;
            for(int k3=0;k3<rd;k3++)for(int k4=0;k4<re;k4++){
                const float2 c=conj(pt.fields[d].e[k3])*pt.fields[e].e[k4];if(zero(c))continue;
                w+=c*entry(P,W,tables+slice+(k3*re+k4)*size+at,C.size,-1);
            }
        };
        if(which==0){for(int di=0;di<F.on;di++){const int d=F.carriers[di];form(d,d,F.p[d][d]);}}
        else if(which==1){for(int ci=0;ci<F.on;ci++){const int c=F.carriers[ci];form(c,c,F.rr_all[c]);}}
        else{const int ei=(which-2)/F.on,ci=(which-2)-ei*F.on;if(ei!=ci){const int e=F.carriers[ei],c=F.carriers[ci];form(e,c,F.p[e][c]);}}
        weights[i]=w;
    }
    __syncwarp();
}
// The fourth order's change of a photon's matrix element along an axis (model.rs `fourth_photons`):
// −C W3 − ½ C R R†R through every ground state b′, W3^c[:, b] = Σ_b′ G_b(ω_c)[Σ_d T^c(E) P^{dd} + Σ_{e≠c} T^e(E) P^{ec}].
__device__ float2 fourth_photon(const Program& P,const Warp& W,const Cluster& C,const Photon& ph,int a,const float2* weights){
    const Fourth& F=P.fourths[C.fourth];const float2* tables=P.tables+C.table;const int n=C.n,c=ph.carrier,size=NG*n;const Point& pt=W.pt;
    int ci=-1;for(int i=0;i<F.on;i++)if(F.carriers[i]==c)ci=i;
    if(ci<0)return zero2();
    float2 z=zero2();
    for(int ei=0;ei<F.on;ei++){
        const int e=F.carriers[ei],at=F.jumps[c][e],re=P.carriers[e].rank;if(at<0)continue;
        const float2* weight=weights+(e==c?0:(2+ei*F.on+ci)*size);
        for(int k=0;k<re;k++){
            const float2 ek=pt.fields[e].e[k];if(zero(ek))continue;float2 s=zero2();
            for(int b=0;b<NG;b++)s+=entry(P,W,tables+at+(((k*3+a)*NG+ph.to)*NG+b)*n+ph.col,C.size,-1)*weight[b*n+ph.col];
            z-=ek*s;
        }
    }
    const int at=F.photons[c],rc=P.carriers[c].rank;const float2* rr=weights+size;
    for(int k=0;k<rc;k++){
        const float2 ck=pt.fields[c].e[k];if(zero(ck))continue;float2 s=zero2();
        for(int b=0;b<NG;b++)s+=entry(P,W,tables+at+((k*3+a)*NG+ph.to)*NG+b,C.size,-1)*rr[b*n+ph.col];
        z-=0.5f*(ck*s);
    }
    return z;
}

// Every photon's matrix element of the cluster's jumps at the point (model.rs `Model::photons`),
// interpolated once per jump: M_a[to, from] = Σ_k E_k J_k − E^D_k J2_k, or the exact responses'
// Σ_l a_l C_a R_l(w_l), per photon and axis (the cluster's photons, all its classes', in order).
__device__ void photon_matrices(const Program& P,const Warp& W,const Cluster& C,int lane,const float2* X,float2* out,float2* weights){
    const float2* tables=P.tables+C.table;const int first=P.classes[C.classes].photons;
    const Class last=P.classes[C.classes+C.nclasses-1];const int count=3*(last.photons+last.nphotons-first);
    if(C.fourth>=0)fourth_weights(P,W,C,lane,weights);
    for(int j=lane;j<count;j+=32){
        const int a=j%3;const Photon ph=P.photons[first+j/3];const Carrier R=P.carriers[ph.carrier];float2 m=zero2();
        if(X==nullptr){
            const Fields& f=W.pt.fields[ph.carrier];const int j1=P.carrier_tables[C.carriers+3*ph.carrier],j2=P.carrier_tables[C.carriers+3*ph.carrier+1];
            #pragma unroll
            for(int k=0;k<MOST_RANK;k++)if(k<R.rank){const int at=((k*3+a)*NG+ph.to)*C.n0+ph.col;m+=f.e[k]*entry(P,W,tables+j1+at,C.size,-1)-f.doppler[k]*entry(P,W,tables+j2+at,C.size,-1);}
        }else for(int b=0;b<R.beams;b++){const int l=P.beam_list[R.first+b];m+=X[exact_at(P.beams[l].near,ph.col,a,ph.to)]*W.pt.a[l];}
        if(C.fourth>=0)m+=fourth_photon(P,W,C,ph,a,weights);
        out[j]=m;
    }
    __syncwarp();
}
// The amplitude u_a of a class at a row of its target cluster (model.rs `Model::photons`): over its
// photons landing in that row, e^{iω̃t} M_a[to, from] ψ[from], M from the photons' matrices; less over its
// explicit states' photons there e^{iω̃t} ψ_e ĉ_a[to] (their dressed photons at the stencil).
__device__ float2 amplitude(const Program& P,const Warp& W,const Cluster& C,const Class& K,int a,int row,double t,const float2* M){
    const int first=P.classes[C.classes].photons,end=P.class_rows[K.rows+row+1];float2 u=zero2();
    for(int i=P.class_rows[K.rows+row];i<end;i++){const Photon ph=P.photons[i];u+=residual(ph.residual,t)*M[3*(i-first)+a]*W.psi[ph.col];}
    for(int i=0;i<K.ndirect;i++){
        const Direct d=P.directs[K.direct+i];if(d.row!=row)continue;
        u-=residual(d.residual,t)*W.psi[d.state]*entry(P,W,P.tables+C.table+C.emission+((d.state-C.n0)*3+a)*NG+d.to,C.size,-1);
    }
    return u;
}

// The picked class's absorbed light along the photon's polarization e (model.rs `Photons::gradient`
// and `Photons::shares`, contracted with e), from one pass over its photons' jump tables. A lane's
// items are (row, carrier) pairs of the target's rows and the class's carriers; of an item, over its
// photons, A_k = Σ e^{iω̃t} ψ[from] Σ_a e_a J_k[a][to, from], B_k the same of J2_k, and the gradient
// Σ_k ∇E_k A_k + E_k Σ_f ∂_f A_k ∇s_f, whose Im⟨f|·⟩ is the exact conditional mean's correction
// (`mean`, still to be divided by ‖f‖²). Each beam l of the class's carriers in their order: its
// share Σ_rows ‖a_l Σ_k c_lk (A_k − w_l B_k)‖² -- beyond the first order the exact responses'
// ‖a_l Σ e^{iω̃t} ψ[from] Σ_a e_a (C_a R_l)[to, from]‖² -- into J.shares and J.beams; returns how many.
__device__ int absorption(const Program& P,Warp& W,const Cluster& C,const Class& K,int nt,const double e[3],double t,const float2* X,int lane,float mean[3]){
    constexpr int ITEMS=(MAXN*MOST_CARRIERS+31)/32;
    const float2* tables=P.tables+C.table;const Point& pt=W.pt;JumpSpace& J=W.jump;const int D=P.focused;
    const float ea[3]={(float)e[0],(float)e[1],(float)e[2]};
    float2 A[ITEMS][MOST_RANK],B[ITEMS][MOST_RANK];int rows[ITEMS],carriers[ITEMS],runs[ITEMS][2];
    #pragma unroll
    for(int x=0;x<3;x++)mean[x]=0.f;
    #pragma unroll 1
    for(int m=0;m<ITEMS;m++){
        const int it=lane+32*m,row=it/P.ncarriers,c=it-row*P.ncarriers;
        const bool mine=row<nt&&((K.carriers>>c)&1);rows[m]=mine?row:-1;carriers[m]=c;
        float2 d[MOST_RANK][MOST_FOCUSED];
        #pragma unroll
        for(int k=0;k<MOST_RANK;k++){
            A[m][k]=B[m][k]=zero2();
            #pragma unroll
            for(int f=0;f<MOST_FOCUSED;f++)d[k][f]=zero2();
        }
        // The item's photons: the carrier's run within the row's.
        int lo=0,hi=0;
        if(mine){
            lo=P.class_rows[K.rows+row];const int end=P.class_rows[K.rows+row+1];
            while(lo<end&&P.photons[lo].carrier<c)lo++;
            hi=lo;while(hi<end&&P.photons[hi].carrier==c)hi++;
        }
        runs[m][0]=lo;runs[m][1]=hi;
        if(!mine)continue;
        const int j1=P.carrier_tables[C.carriers+3*c],j2=P.carrier_tables[C.carriers+3*c+1],rank=P.carriers[c].rank;
        for(int i=lo;i<hi;i++){
            const Photon ph=P.photons[i];
            const float2 phase=residual(ph.residual,t)*W.psi[ph.col];
            #pragma unroll
            for(int k=0;k<MOST_RANK;k++)if(k<rank){
                #pragma unroll 1
                for(int a=0;a<3;a++){
                    const int at=((k*3+a)*NG+ph.to)*C.n0+ph.col;const float2 pe=ea[a]*phase;
                    float2 slope[MOST_FOCUSED];const float2 value=entry_slopes(P,W,tables+j1+at,C.size,slope);
                    A[m][k]+=pe*value;B[m][k]+=pe*entry(P,W,tables+j2+at,C.size,-1);
                    #pragma unroll
                    for(int f=0;f<MOST_FOCUSED;f++)if(f<D)d[k][f]+=pe*slope[f];
                }
            }
        }
        const Fields& fl=pt.fields[c];const float2 fr=J.f[row];
        #pragma unroll
        for(int x=0;x<3;x++){
            float2 g=zero2();
            #pragma unroll
            for(int k=0;k<MOST_RANK;k++)if(k<rank){
                g+=fl.grad[k][x]*A[m][k];
                #pragma unroll
                for(int f=0;f<MOST_FOCUSED;f++)if(f<D)g+=pt.grad_s[f][x]*(fl.e[k]*d[k][f]);
            }
            mean[x]+=fr.x*g.y-fr.y*g.x;
        }
    }
    #pragma unroll
    for(int x=0;x<3;x++)mean[x]=warp_sum(mean[x]);
    int count=0;
    for(int c=0;c<P.ncarriers;c++){
        if(!((K.carriers>>c)&1))continue;const Carrier R=P.carriers[c];
        for(int j=0;j<R.beams;j++){
            const int l=P.beam_list[R.first+j];const Beam& b=P.beams[l];const float2 al=pt.a[l];const float wl=pt.shift[l];
            float v=0.f;
            #pragma unroll
            for(int m=0;m<ITEMS;m++)if(rows[m]>=0&&carriers[m]==c){
                float2 g=zero2();
                if(X==nullptr){
                    #pragma unroll
                    for(int k=0;k<MOST_RANK;k++)if(k<R.rank)g+=make_float2(b.coefficients[k][0],b.coefficients[k][1])*(A[m][k]-wl*B[m][k]);
                }else for(int i=runs[m][0];i<runs[m][1];i++){
                    const Photon ph=P.photons[i];float2 z=zero2();
                    #pragma unroll
                    for(int a=0;a<3;a++)z+=ea[a]*X[exact_at(b.near,ph.col,a,ph.to)];
                    g+=residual(ph.residual,t)*W.psi[ph.col]*z;
                }
                v+=norm2(al*g);
            }
            const float w=warp_sum(v);if(lane==0){J.shares[count]=w;J.beams[count]=l;}count++;
        }
    }
    __syncwarp();
    // The explicit states' part of the photon (`Photons::absorption`): its weight along e, Σ_rows |Σ e^{iω̃t}
    // ψ_e Σ_a e_a ĉ_a[to]|², shared by the beams in proportion to their drives into the explicit states,
    // Σ |a_l Σ_k c_lk U_k ψ_b|² over the kept drives -- added to a beam's share, or a share of its own after them.
    if(K.ndirect>0){
        float2 v=zero2();
        if(lane<nt)for(int i=0;i<K.ndirect;i++){
            const Direct d=P.directs[K.direct+i];if(d.row!=lane)continue;float2 z=zero2();
            for(int a=0;a<3;a++)z+=ea[a]*entry(P,W,tables+C.emission+((d.state-C.n0)*3+a)*NG+d.to,C.size,-1);
            v+=residual(d.residual,t)*W.psi[d.state]*z;
        }
        const float total=warp_sum(lane<nt?norm2(v):0.f);
        // Per beam its drives' Σ |a_l Σ_k c_lk U_k ψ_b|², in the drives' and their carriers' beams' order.
        int beams[MOST_BEAMS];float drive[MOST_BEAMS];int nb=0;float sum=0.f;
        for(int i=0;i<C.nterms;i++){
            const Term T=P.terms[C.terms+i];if(T.kind!=0)continue;const Carrier Rc=P.carriers[T.carrier];
            for(int j=0;j<Rc.beams;j++){
                const int l=P.beam_list[Rc.first+j];const Beam& b=P.beams[l];float2 z=zero2();
                for(int k=0;k<Rc.rank;k++)z+=make_float2(b.coefficients[k][0],b.coefficients[k][1])*make_float2(T.u[k][0],T.u[k][1]);
                const float d=norm2(z*pt.a[l]*W.psi[T.col]);int at=0;while(at<nb&&beams[at]!=l)at++;
                if(at==nb){beams[nb]=l;drive[nb]=0.f;nb++;}
                drive[at]+=d;sum+=d;
            }
        }
        if(sum>0.f){
            for(int i=0;i<nb;i++){
                const float share=total*drive[i]/sum;int at=0;while(at<count&&J.beams[at]!=beams[i])at++;
                __syncwarp();if(lane==0){if(at<count)J.shares[at]+=share;else{J.shares[count]=share;J.beams[count]=beams[i];}}__syncwarp();
                if(at==count)count++;
            }
        }
    }
    return count;
}

// The fourth order's start from a bare ground state after a jump (engine.rs `slow_start`; the initial
// states have theirs from the CPU): ψ ← Π f/√A, the threshold θ ← θ/A, A = ‖Π f‖², Π = 1 + ½R†R + YW at
// the point in the cluster's frames -- A < 1: the transient emits at once with probability 1 − A (θ/A
// ≥ 1, a jump at the step's start); A > 1: the dressing's rebuild delays the next jump. No draws.
__device__ void slow_start(const Program& P,Warp& W,Traj& t,int lane){
    const Cluster C=P.clusters[t.cluster];if(C.fourth<0)return;
    const Fourth& F=P.fourths[C.fourth];const Entries E(C.n,lane);float2 pi[SLOTS];
    #pragma unroll
    for(int j=0;j<SLOTS;j++)pi[j]=zero2();
    fourth_quadratic(P,W,C,F,F.rr,0.5f,lane,pi);fourth_quadratic(P,W,C,F,F.yw,1.f,lane,pi);
    in_frames(P,C,E,t.time,lane,pi);
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane))W.m[0][lane+32*j]=pi[j]+make_float2(E.row[j]==E.col[j]?1.f:0.f,0.f);
    __syncwarp();
    const float a=apply(W.m[0],W.psi,C.n,lane);
    t.threshold/=(double)a;t.log_threshold=log(t.threshold);
}

// A real orthonormal pair across the unit vector n (engine.rs `across`).
__device__ void across(const double n[3],double pair[2][3]){
    double a[3]={0.,0.,0.};
    if(fabs(n[0])<=fabs(n[1])&&fabs(n[0])<=fabs(n[2]))a[0]=1.;else if(fabs(n[1])<=fabs(n[2]))a[1]=1.;else a[2]=1.;
    double e1[3]={n[1]*a[2]-n[2]*a[1],n[2]*a[0]-n[0]*a[2],n[0]*a[1]-n[1]*a[0]};
    const double l=sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]);
    for(int i=0;i<3;i++){e1[i]/=l;pair[0][i]=e1[i];}
    pair[1][0]=n[1]*e1[2]-n[2]*e1[1];pair[1][1]=n[2]*e1[0]-n[0]*e1[2];pair[1][2]=n[0]*e1[1]-n[1]*e1[0];
}

// A quantum jump at the trajectory's position and time (engine.rs `Engine::jump`), with the exact
// response when `exact`: the class and the cluster it lands in, the photon's direction from the
// dipole pattern, its polarization, the recoils, and the state the photon leaves.
// A jump's first draws (engine.rs `jump`), each class's weight in J.weights and their total positive:
// the class, the next threshold, the polarization's and the absorbed beam's uniforms -- before the
// direction's rejection loop, so that a twin on the same stream keeps its thresholds and choices.
struct Draws{int pick;double threshold,polarization,absorbed;};
__device__ Draws draw(Rng& rng,const JumpSpace& J,int nclasses,double total){
    Draws d;double x=rng.uniform()*total;d.threshold=rng.open_uniform();d.polarization=rng.uniform();d.absorbed=rng.uniform();
    d.pick=nclasses-1;for(int c=0;c<nclasses;c++){const double w=(double)J.weights[c];if(x<w){d.pick=c;break;}x-=w;}
    return d;
}
// The photon of the picked class's amplitudes J.u (nt rows): its direction n from the dipole pattern
// tr G − n·Re(G)·n of G_ab = ⟨u_a|u_b⟩, its polarization e across it from ‖Σ_a e_a u_a‖² on a real
// orthonormal pair, f = Σ_a e_a u_a into J.f; returns ‖f‖².
__device__ float emit(Rng& rng,JumpSpace& J,int nt,double polarization,int lane,double n[3],double e[3]){
    double gram[3][3];
    for(int a=0;a<3;a++)for(int b=0;b<3;b++){float s=0.f;for(int i=0;i<nt;i++){const float2 x=J.u[a][i],y=J.u[b][i];s+=x.x*y.x+x.y*y.y;}gram[a][b]=(double)s;}
    const double trace=gram[0][0]+gram[1][1]+gram[2][2];
    for(;;){
        rng.direction(n);double q=0.;for(int a=0;a<3;a++)for(int b=0;b<3;b++)q+=n[a]*gram[a][b]*n[b];
        if(rng.uniform()*trace<=trace-q)break;
    }
    double pair[2][3];across(n,pair);float weights[2];float2 f[2]={zero2(),zero2()};
    for(int j=0;j<2;j++){
        if(lane<nt)for(int a=0;a<3;a++)f[j]+=(float)pair[j][a]*J.u[a][lane];
        weights[j]=warp_sum(lane<nt?norm2(f[j]):0.f);
    }
    const int chosen=polarization*((double)weights[0]+(double)weights[1])<(double)weights[0]?0:1;
    for(int a=0;a<3;a++)e[a]=pair[chosen][a];
    if(lane<nt)J.f[lane]=f[chosen];
    __syncwarp();
    return weights[chosen];
}

__device__ __noinline__ void transient(const Program& P,const Run& R,Warp& W,Traj& t,const double u[4],Rng& rng,int lane,float2* E);
__device__ float2 response(const Program& P,const Warp& W,const Cluster& C,int c,int i,int col);
__device__ __noinline__ void jump(const Program& P,const Run& R,Warp& W,Traj& t,bool exact,int lane,float2* scratch){
    t.jumps++;Rng rng(R.seed,PURPOSE_JUMP,t.index,(u32)t.jumps);
    point(P,W,t,lane);
    const Cluster C=P.clusters[t.cluster];
    {
        float s=0.f;for(int b=0;b<C.n;b++)s+=norm2(W.psi[b]);const float scale=1.f/sqrtf(s);
        __syncwarp();if(lane<C.n)W.psi[lane]=scale*W.psi[lane];__syncwarp();
    }
    SPAN_BEGIN(responses);
    if(exact)COUNT(COUNT_EXACT);
    const float2* X=exact&&exact_responses(P,W,C,lane,scratch,scratch+P.rows_at)?scratch:nullptr;
    SPAN_END(SPAN_EXACT,responses);SPAN_BEGIN(photons);
    float2* M=scratch+P.nnear*MAXN*3*NG;photon_matrices(P,W,C,lane,X,M,scratch+P.weights);
    JumpSpace& J=W.jump;
    // The class, and the cluster it lands the atom in: by Σ_a ‖u_a‖².
    double total=0.;
    for(int c=0;c<C.nclasses;c++){
        const Class K=P.classes[C.classes+c];const int nt=P.clusters[K.target].n0;
        float v=0.f;if(lane<3*nt)v=norm2(amplitude(P,W,C,K,lane%3,lane/3,t.time,M));
        const float w=warp_sum(v);if(lane==0)J.weights[c]=w;total+=(double)w;
    }
    __syncwarp();
    SPAN_END(SPAN_PHOTONS,photons);
    if(!(total>0.)){t.threshold=rng.open_uniform();t.log_threshold=log(t.threshold);return;}
    const Draws d=draw(rng,J,C.nclasses,total);
    const Class K=P.classes[C.classes+d.pick];const Cluster T=P.clusters[K.target];const int nt=T.n0;
    // Into a cluster with explicit states its transient's four draws, before the directions' loops.
    const bool into=T.n>T.n0;double tu[4];if(into)for(int i=0;i<4;i++)tu[i]=rng.uniform();
    if(lane<3*nt)J.u[lane%3][lane/3]=amplitude(P,W,C,K,lane%3,lane/3,t.time,M);
    __syncwarp();
    double n[3],e[3];const float size=emit(rng,J,nt,d.polarization,lane,n,e);
    // The recoils: the photon's, the absorbed beam's and the exact conditional mean's correction.
    double kick[3];for(int a=0;a<3;a++)kick[a]=-K.wave_number*n[a];
    SPAN_BEGIN(absorbed);
    float m[3];const int count=absorption(P,W,C,K,nt,e,t.time,X,lane,m);
    SPAN_END(SPAN_ABSORPTION,absorbed);
    double mean[3],sum=0.;
    for(int x=0;x<3;x++)mean[x]=(double)m[x]/(double)size;
    for(int i=0;i<count;i++)sum+=(double)J.shares[i];
    if(sum>0.){
        double x=d.absorbed*sum;int drawn=J.beams[count-1];
        for(int i=0;i<count;i++){const double w=(double)J.shares[i];if(x<w){drawn=J.beams[i];break;}x-=w;}
        double average[3]={0.,0.,0.};
        for(int i=0;i<count;i++){const Beam& b=P.beams[J.beams[i]];for(int a=0;a<3;a++)average[a]+=(double)J.shares[i]*((double)b.direction[a]*(double)b.k);}
        const Beam& b=P.beams[drawn];
        for(int a=0;a<3;a++)kick[a]+=(double)b.direction[a]*(double)b.k+mean[a]-average[a]/sum;
    }else for(int a=0;a<3;a++)kick[a]+=mean[a];
    #pragma unroll
    for(int i=0;i<3;i++)t.p[i]+=P.axes[i][0]*kick[0]+P.axes[i][1]*kick[1]+P.axes[i][2]*kick[2];
    const float scale=1.f/sqrtf(size);
    __syncwarp();if(lane<MAXM)W.psi[lane]=lane<nt?scale*J.f[lane]:zero2();__syncwarp();
    t.cluster=K.target;t.logn=0.;t.threshold=d.threshold;t.log_threshold=log(d.threshold);
    if(into)transient(P,R,W,t,tu,rng,lane,scratch+P.transient_at);
    slow_start(P,W,t,lane);
}

// ---------------------------------------------------------------------------------------------
// The held twin (control.rs, engine.rs `Engine::held`): the trajectory's internal state on its own
// streams at the twin's point, its generator and jumps there computed once; it does not move.

// The twin's flow for τ without jumps: ψ ← U ψ twice, U the half step's propagator for a whole step,
// exp(−i H τ/2) by the exponentials' series for a step's part; ln‖ψ‖² at the middle (`lh`) and the end.
__device__ void hold_advance(const Program& P,const Run& R,Warp& W,Twin& h,double tau,int lane,double& lh){
    const int n=P.clusters[h.cluster].n,at=R.held.blocks[h.cluster];const float2* U=R.held.u+at;
    if(tau!=P.dt){
        const Entries E(n,lane);
        __syncwarp();
        #pragma unroll
        for(int j=0;j<SLOTS;j++)if(E.has(j,lane))W.m[0][lane+32*j]=R.held.h[at+lane+32*j];
        __syncwarp();
        exponential(P,W,E,0.5*tau,lane);U=W.m[2];
    }
    const float nh=apply(U,W.twin,n,lane);lh=h.logn+(double)logf(nh);
    const float n1=apply(U,W.twin,n,lane);h.logn=lh+(double)logf(n1);
}

// The twin's jump (engine.rs `Engine::jump` of a held atom): the trajectory's stream of the twin's
// jump number, the classes' amplitudes from the jump operators at its point, the same draws as a
// moving atom's, no kicks.
__device__ void hold_jump(const Program& P,const Run& R,Warp& W,Twin& h,u32 index,int lane){
    h.jumps++;Rng rng(R.seed,PURPOSE_JUMP,index,(u32)h.jumps);
    const Cluster C=P.clusters[h.cluster];JumpSpace& J=W.jump;
    // u_x = L_x ψ of a class at the lane's (x, row).
    auto amplitude=[&](int c,int nt)->float2{
        float2 u=zero2();if(lane>=3*nt)return u;
        const int x=lane%3,row=lane/3;const float2* l=R.held.l+R.held.jumps[C.classes+c]+(x*nt+row)*C.n;
        for(int j=0;j<C.n;j++)u+=l[j]*W.twin[j];
        return u;
    };
    double total=0.;
    for(int c=0;c<C.nclasses;c++){
        const int nt=P.clusters[P.classes[C.classes+c].target].n;
        const float w=warp_sum(norm2(amplitude(c,nt)));if(lane==0)J.weights[c]=w;total+=(double)w;
    }
    __syncwarp();
    if(!(total>0.)){h.threshold=rng.open_uniform();h.log_threshold=log(h.threshold);return;}
    const Draws d=draw(rng,J,C.nclasses,total);
    const Class K=P.classes[C.classes+d.pick];const int nt=P.clusters[K.target].n;
    const float2 u=amplitude(d.pick,nt);if(lane<3*nt)J.u[lane%3][lane/3]=u;
    __syncwarp();
    double n[3],e[3];const float size=emit(rng,J,nt,d.polarization,lane,n,e);
    const float scale=1.f/sqrtf(size);
    __syncwarp();if(lane<MAXN)W.twin[lane]=lane<nt?scale*J.f[lane]:zero2();__syncwarp();
    h.cluster=K.target;h.logn=0.;h.threshold=d.threshold;h.log_threshold=log(d.threshold);
}

// One step of the twin (engine.rs `Engine::step` of a held atom): where ‖ψ‖² crosses the threshold
// the step is redone to the crossing, the jump made there and the rest taken from the state it left.
__device__ void hold_step(const Program& P,const Run& R,Warp& W,Twin& h,u32 index,int lane){
    double left=P.dt;
    for(;;){
        const Twin start=h;
        if(lane<MAXN)W.start[lane]=W.twin[lane];
        double lh;hold_advance(P,R,W,h,left,lane,lh);
        if(h.logn>h.log_threshold)break;
        double ta,tb,la,lb;
        if(lh<=h.log_threshold){ta=0.;tb=0.5*left;la=start.logn;lb=lh;}else{ta=0.5*left;tb=left;la=lh;lb=h.logn;}
        const double at=lb<la?fmin(fmax(ta+(h.log_threshold-la)/(lb-la)*(tb-ta),ta),tb):tb;
        h=start;__syncwarp();if(lane<MAXN)W.twin[lane]=W.start[lane];__syncwarp();
        if(at>0.){double unused;hold_advance(P,R,W,h,at,lane,unused);}
        hold_jump(P,R,W,h,index,lane);
        left-=at;
        if(!(left>0.))break;
    }
}

// The twin's record of an output (control.rs `Twin::populations`, stats.rs `Group::add`): its 32
// populations c_k at its point -- the readout's map of its density ψψ†/‖ψ‖² -- and Σ_k c_k², Σ_k c_k p_k
// with the atom's populations p (already in `out`, zero when lost).
__device__ void hold_record(const Program& P,const Run& R,Warp& W,const Twin& h,int lane,double* out){
    // In double precision from ψ as it is, the map's coefficients the CPU's: Σ_k c_k is one to rounding.
    const int n=P.clusters[h.cluster].n;const double* map=R.held.readout+32*R.held.blocks[h.cluster];
    double norm=0.;for(int b=0;b<n;b++)norm+=dnorm2(d2(W.twin[b]));
    double c=0.;
    for(int i=0;i<n;i++){
        const double2 pi=d2(W.twin[i]);c+=dnorm2(pi)*map[32*(i*n+i)+lane];
        for(int j=i+1;j<n;j++){const double2 r=dmul(pi,d2(conj(W.twin[j])));c+=r.x*map[32*(i*n+j)+lane]+r.y*map[32*(j*n+i)+lane];}
    }
    c/=norm;
    out[TWIN+lane]=c;
    const double cc=warp_sum(c*c),cy=warp_sum(c*out[3+lane]);
    if(lane==0){out[TWIN+32]=cc;out[TWIN+33]=cy;}
}

// ---------------------------------------------------------------------------------------------
// The step.

// The trap's potential of ψ at the point (model.rs `trap_potential`, with `bound` its upper bound
// `trap_bound`): Re⟨ψ|·|ψ⟩/⟨ψ|ψ⟩ of h0 at the point less with the trap beam off, and of −Φ S less
// −Φ_off S with the trap beam off -- Φ_off of the fields with the trap beam's own part a c_k taken
// out of its carrier's, so that a tweezer that is a near carrier is in the depth as a folded one
// is; the bound takes twice the light shift's bound for the latter.
__device__ float trap(const Program& P,const Warp& W,const Cluster& C,const Entries& E,const float2 (&h0)[SLOTS],double t,bool bound,int lane){
    const float2* tables=P.tables+C.table;const Point& pt=W.pt;
    float2 difference[SLOTS],far[SLOTS],slope[MOST_FOCUSED][SLOTS];
    gather<false>(P,W,true,tables+C.h0,C.size,E.nn,lane,far,slope);
    #pragma unroll
    for(int j=0;j<SLOTS;j++)difference[j]=h0[j]-far[j];
    if(bound)return quadratic(W,E,difference,lane)+P.light2;
    // The fields with the trap's light off (its beams' parts out of their carriers'): pt.off.
    float2 here[SLOTS];
    // The sparse groups' entries at the lane's places (their adjoints none: the form takes a pair
    // standing for two twice).
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(C.nbeats>0&&E.has(j,lane)){
        const int e=lane+32*j,end=P.item_first[C.positions+e+1];
        for(int i=P.item_first[C.positions+e];i<end;i++){
            const int4 it=__ldg(P.items+i);if((it.z>>11)&1)continue;
            const int k1=it.z&3,k=(it.z>>2)&3,ca=(it.z>>4)&7,ce=(it.z>>7)&7;const float both=(it.z>>10)&1?2.f:1.f;
            const float2 fa=pt.fields[ca].e[k],fe=pt.fields[ce].e[k1],oa=pt.off[ca][k],oe=pt.off[ce][k1];
            const float2 phase=residual(P.residuals[C.beats+(it.z>>12)],t),c=zero2()-both*(conj(fe)*fa)*phase,c0=both*(conj(oe)*oa)*phase;
            const float2* x=tables+it.x;
            difference[j]+=c*entry(P,W,x,C.size,-1)+c0*far_entry(P,W,x,C.size);
        }
    }
    for(int gi=0;gi<C.ngroups;gi++){
        const Group G=P.groups[C.groups+gi];if(G.count!=E.nn)continue;
        const Fields& fa=pt.fields[G.absorbed];const Fields& fe=pt.fields[G.emitted];
        const float2 phase=residual(G.residual,t);
        for(int pi=0;pi<G.npairs;pi++){
            const int2 pair=P.pairs[G.pairs+pi];const int k1=pair.x,k=pair.y;const float both=G.hermitian&&k1<k?2.f:1.f;
            const float2 oa=pt.off[G.absorbed][k],oe=pt.off[G.emitted][k1];
            const float2 c=zero2()-both*(conj(fe.e[k1])*fa.e[k])*phase,c0=both*(conj(oe)*oa)*phase;
            gather<false>(P,W,false,tables+G.s+pi*E.nn,C.size,E.nn,lane,here,slope);
            gather<false>(P,W,true,tables+G.s+pi*E.nn,C.size,E.nn,lane,far,slope);
            #pragma unroll
            for(int j=0;j<SLOTS;j++)difference[j]+=c*here[j]+c0*far[j];
        }
    }
    return quadratic(W,E,difference,lane);
}

// ---------------------------------------------------------------------------------------------
// The exact step (engine.rs # The exact step): a cluster with explicit excited states or beats takes
// its internal state exactly through the step at the half step's point -- multirate: the motion's step
// keeps its own size --, every jump at the root of ‖ψ(t)‖² = θ, the impulse ∫⟨F⟩dt along it.
//
// Its vectors are lane-replicated: lane L holds entry L & 15 (n ≤ MAXM), the warp's halves alike; an
// n × n matrix's row r is lanes r's (columns [0, 8)) and r + 16's (columns [8, 16)). A segment's matrices
// lie in the trajectory's scratch from P.exact_at (`exact_segment`), Q parts and nn = n² entries each: per
// part q of its components (0 the static one, 1 + j beat j's) H_q at q·nn and the gradients' Hermitian
// parts G_q per axis at (Q + 3q + a)·nn; the components' Γ_q from 4Q·nn; the static path's propagator
// exp(−iH₀δ) at 5Q·nn and its decay i(H₀ − H₀†) next; two working matrices; the parts' phases at the
// step's start (Q) and their norms (`exact_norms`).

__device__ __forceinline__ float2 shfl2(float2 v,int from){return make_float2(__shfl_sync(FULL,v.x,from),__shfl_sync(FULL,v.y,from));}
__device__ __forceinline__ float2 halves(float2 v){return v+make_float2(__shfl_xor_sync(FULL,v.x,16),__shfl_xor_sync(FULL,v.y,16));}
// The lane's half of x: x_b, b ∈ [8h, 8h + 8).
// Through the warp's shared vector: lanes 0..15 write theirs, each half of the warp reads its own (one
// address a load, the half's lanes alike).
__device__ __forceinline__ void half_of(float2 x,int lane,float2 (&xb)[8],float4* v){
    __syncwarp();if(lane<16)((float2*)v)[lane]=x;__syncwarp();
    const int h=lane>>4;
    #pragma unroll
    for(int j=0;j<4;j++){const float4 z=v[4*h+j];xb[2*j]=make_float2(z.x,z.y);xb[2*j+1]=make_float2(z.z,z.w);}
}
// The lane's row-half of an n × n matrix (row-major), zero past n.
__device__ __forceinline__ void row_half(const float2* A,int n,int lane,float2 (&a)[8]){
    const int r=lane&15,h=lane>>4;
    #pragma unroll
    for(int j=0;j<8;j++){const int b=8*h+j;a[j]=r<n&&b<n?A[r*n+b]:zero2();}
}
// (A x)_r, lane-replicated, of A's row-half and x's half.
__device__ __forceinline__ float2 times(const float2 (&a)[8],const float2 (&xb)[8]){
    float2 s=zero2();
    #pragma unroll
    for(int j=0;j<8;j++)s+=a[j]*xb[j];
    return halves(s);
}
// The lane's part of ⟨x|A|x⟩ (its row's half), x̄ the lane's entry conjugated.
__device__ __forceinline__ float2 form_part(const float2 (&a)[8],const float2 (&xb)[8],float2 xr){
    float2 s=zero2();
    #pragma unroll
    for(int j=0;j<8;j++)s+=a[j]*xb[j];
    return conj(xr)*s;
}
// ‖x‖² of a lane-replicated vector.
__device__ __forceinline__ float vnorm(float2 x,int lane){return warp_sum(lane<16?norm2(x):0.f);}

// x ← exp(−iBτ)x for B's row-halves (engine.rs `propagate_dyn`): sub-steps of ‖B‖₁τ ≤ ½ (`norm` bounds
// ‖B‖₁), each through the term whose bound falls below its share of single precision's round-off.
__device__ float2 taylor(const float2 (&b)[8],float norm,float tau,float2 x,int lane,float4* v){
    const float y0=norm*fabsf(tau);const int parts=max((int)ceilf(2.f*y0),1);const float step=tau/(float)parts,y=y0/(float)parts,share=EXACT_ROUND/(float)parts;
    int terms=1;float size=y;while(size>share&&terms<40){terms++;size*=y/(float)terms;}
    for(int p=0;p<parts;p++){
        float2 term=x,sum=x;
        for(int k=1;k<=terms;k++){float2 xb[8];half_of(term,lane,xb,v);const float2 v=times(b,xb);const float c=step/(float)k;term=make_float2(v.y*c,-v.x*c);sum+=term;}
        x=sum;
    }
    return x;
}

// A cluster's segment of an exact step (engine.rs `Segment`): its size and parts, the parts it keeps (bit
// q; bit 0 the static one), its grid's spacing from the step's start, its path -- beating (CF4 over each
// spacing) or static (the propagator over a whole spacing, the series on the vector off the grid) --, and
// in the scratch its parts' phases at the step's start, their norms (`exact_norms`) and their Ω_q. Its
// times are single precision from the step's start (a step's τ holds them to 1e-7 of itself).
struct Segment{int n,parts;unsigned kept;int beating;double start,spacing;float delta;const float2* base;const float* F;const float* w;float4* v;};

// A kept beat's phase e^{iΩ_q t}: its phase at the step's start (double precision) times the rest's.
__device__ __forceinline__ float2 beat_phase(const Segment& S,int q,float t){float s,c;sincosf(S.w[q]*t,&s,&c);return S.base[q]*make_float2(c,s);}

// x over [t, t + len) by CF4 (engine.rs `cf4`, `Segment::combine`): exp(−i len B₂) exp(−i len B₁) x,
// B₁ = w₁H(t₁) + w₂H(t₂), B₂ = w₂H(t₁) + w₁H(t₂) at the Gauss nodes t_i = t + c_i len.
__device__ float2 cf4(const Segment& S,const float2* X,float t,float len,float2 x,int lane){
    const float r=0.28867513459481287f,w1=0.25f+r,w2=0.25f-r;const int nn=S.n*S.n;
    const float t1=t+(0.5f-r)*len,t2=t+(0.5f+r)*len;
    float2 b1[8],b2[8];row_half(X,S.n,lane,b1);
    #pragma unroll
    for(int j=0;j<8;j++){b1[j]=0.5f*b1[j];b2[j]=b1[j];}
    float n1=0.5f*S.F[0],n2=n1;
    for(int q=1;q<S.parts;q++){
        if(!((S.kept>>q)&1u))continue;
        const float2 p1=beat_phase(S,q,t1),p2=beat_phase(S,q,t2),z1=w1*p1+w2*p2,z2=w2*p1+w1*p2;
        float2 a[8];row_half(X+q*nn,S.n,lane,a);
        #pragma unroll
        for(int j=0;j<8;j++){b1[j]+=z1*a[j];b2[j]+=z2*a[j];}
        n1+=sqrtf(norm2(z1))*S.F[q];n2+=sqrtf(norm2(z2))*S.F[q];
    }
    x=taylor(b1,n1,len,x,lane,S.v);
    return taylor(b2,n2,len,x,lane,S.v);
}

// x over [t, t + len) by a segment (engine.rs `Segment::advance`): CF4 where it beats; else the series of
// exp(−iH₀ len) on the vector -- the CPU's propagator over a whole spacing, as exact: a single-precision
// propagator applied at every spacing would compound its own rounding over the step (1e-4 of ln‖ψ‖² over 300
// spacings), the series' rounding is new at each.
__device__ float2 seg_advance(const Segment& S,const float2* X,float t,float len,bool node,float2 x,int lane){
    if(S.beating)return cf4(S,X,t,len,x,lane);
    float2 a[8];row_half(X,S.n,lane,a);return taylor(a,S.F[0],len,x,lane,S.v);
}

// The conditional force's lab components at the time t (engine.rs `Segment::force`): −Re⟨x|G(t)|x⟩/⟨x|x⟩,
// G(t) = G₀ + 2 Re Σ e^{iΩt} G_Ω over the kept positive beats; returns ⟨x|x⟩.
__device__ float seg_force(const Segment& S,const float2* X,float t,float2 x,int lane,float f[3]){
    const int Q=S.parts,nn=S.n*S.n,half=(Q-1)/2;float2 xb[8];half_of(x,lane,xb,S.v);float s[3];
    #pragma unroll
    for(int a=0;a<3;a++){float2 g[8];row_half(X+(Q+a)*nn,S.n,lane,g);s[a]=form_part(g,xb,x).x;}
    for(int q=1+half;q<Q;q++){
        if(!((S.kept>>q)&1u))continue;
        const float2 p=2.f*beat_phase(S,q,t);
        #pragma unroll
        for(int a=0;a<3;a++){float2 g[8];row_half(X+(Q+3*q+a)*nn,S.n,lane,g);s[a]+=(p*form_part(g,xb,x)).x;}
    }
    const float norm=vnorm(x,lane);
    #pragma unroll
    for(int a=0;a<3;a++)f[a]=-warp_sum(s[a])/norm;
    return norm;
}

// A panel's forces at its middle and its end (`seg_force` of each): one pass over the G parts.
__device__ void seg_forces(const Segment& S,const float2* X,float t1,float t2,float2 x1,float2 x2,int lane,float f1[3],float f2[3],float& n1,float& n2){
    const int Q=S.parts,nn=S.n*S.n,half=(Q-1)/2;float2 xb1[8],xb2[8];half_of(x1,lane,xb1,S.v);half_of(x2,lane,xb2,S.v);float s1[3],s2[3];
    #pragma unroll
    for(int a=0;a<3;a++){float2 g[8];row_half(X+(Q+a)*nn,S.n,lane,g);s1[a]=form_part(g,xb1,x1).x;s2[a]=form_part(g,xb2,x2).x;}
    for(int q=1+half;q<Q;q++){
        if(!((S.kept>>q)&1u))continue;
        const float2 p1=2.f*beat_phase(S,q,t1),p2=2.f*beat_phase(S,q,t2);
        #pragma unroll
        for(int a=0;a<3;a++){float2 g[8];row_half(X+(Q+3*q+a)*nn,S.n,lane,g);s1[a]+=(p1*form_part(g,xb1,x1)).x;s2[a]+=(p2*form_part(g,xb2,x2)).x;}
    }
    n1=vnorm(x1,lane);n2=vnorm(x2,lane);
    #pragma unroll
    for(int a=0;a<3;a++){f1[a]=-warp_sum(s1[a])/n1;f2[a]=-warp_sum(s2[a])/n2;}
}
// A panel's two spacings by CF4 (`cf4` twice) with the parts loaded once: both spacings' B₁ and B₂.
__device__ void cf4_panel(const Segment& S,const float2* X,float t,float len,float2 x,int lane,float2& middle,float2& last){
    const float r=0.28867513459481287f,w1=0.25f+r,w2=0.25f-r;const int nn=S.n*S.n;
    const float ta=t+(0.5f-r)*len,tb=t+(0.5f+r)*len;
    float2 a1[8],a2[8],b1[8],b2[8];row_half(X,S.n,lane,a1);
    #pragma unroll
    for(int j=0;j<8;j++){a1[j]=0.5f*a1[j];a2[j]=a1[j];b1[j]=a1[j];b2[j]=a1[j];}
    float na1=0.5f*S.F[0],na2=na1,nb1=na1,nb2=na1;
    for(int q=1;q<S.parts;q++){
        if(!((S.kept>>q)&1u))continue;
        const float2 p1=beat_phase(S,q,ta),p2=beat_phase(S,q,tb),p3=beat_phase(S,q,ta+len),p4=beat_phase(S,q,tb+len);
        const float2 za1=w1*p1+w2*p2,za2=w2*p1+w1*p2,zb1=w1*p3+w2*p4,zb2=w2*p3+w1*p4;
        float2 h[8];row_half(X+q*nn,S.n,lane,h);
        #pragma unroll
        for(int j=0;j<8;j++){a1[j]+=za1*h[j];a2[j]+=za2*h[j];b1[j]+=zb1*h[j];b2[j]+=zb2*h[j];}
        const float f=S.F[q];na1+=sqrtf(norm2(za1))*f;na2+=sqrtf(norm2(za2))*f;nb1+=sqrtf(norm2(zb1))*f;nb2+=sqrtf(norm2(zb2))*f;
    }
    x=taylor(a1,na1,len,x,lane,S.v);middle=taylor(a2,na2,len,x,lane,S.v);
    x=taylor(b1,nb1,len,middle,lane,S.v);last=taylor(b2,nb2,len,x,lane,S.v);
}
// A panel's two halves from x (`seg_advance` twice): a beating segment's by `cf4_panel`, a static one's
// whole spacings by its propagator loaded once.
__device__ void seg_advances(const Segment& S,const float2* X,float t,float half,bool whole,float2 x,int lane,float2& middle,float2& last){
    if(S.beating){cf4_panel(S,X,t,half,x,lane,middle,last);return;}
    float2 a[8];row_half(X,S.n,lane,a);middle=taylor(a,S.F[0],half,x,lane,S.v);last=taylor(a,S.F[0],half,middle,lane,S.v);
}

// −d‖x‖²/dt at the time t (engine.rs `Segment::rate`): static ⟨x|D|x⟩, D = i(H₀ − H₀†); beating −2 Im⟨x|H(t)|x⟩.
__device__ float seg_rate(const Segment& S,const float2* X,float t,float2 x,int lane){
    const int Q=S.parts,nn=S.n*S.n;float2 xb[8];half_of(x,lane,xb,S.v);float2 a[8];
    if(!S.beating){row_half(X+(5*Q+1)*nn,S.n,lane,a);return warp_sum(form_part(a,xb,x).x);}
    row_half(X,S.n,lane,a);float2 z=form_part(a,xb,x);
    for(int q=1;q<Q;q++){if(!((S.kept>>q)&1u))continue;row_half(X+q*nn,S.n,lane,a);z+=beat_phase(S,q,t)*form_part(a,xb,x);}
    return -2.f*warp_sum(z.y);
}

// A group's term at the table place `at` (its S at s + at, its Q at q + at) into the lane's entry e of part
// p, or its adjoint (model.rs `components`' `scatter`): ee S into M (H_p until the end), ee Q into Γ_p,
// −(c S + ee Σ_f ∇s_f ∂_f S) into G_p per axis.
__device__ __forceinline__ void group_term(const Program& P,const Warp& W,const Cluster& C,const float2* s,const float2* q,float2 ee,const float2 (&c)[3],
        int p,bool adjoint,int e,float2* X,float2* G,float2* Gamma,int nn){
    const Point& pt=W.pt;float2 ds[MOST_FOCUSED];const float2 sv=entry_slopes(P,W,s,C.size,ds),qv=entry(P,W,q,C.size,-1);
    float2 h=ee*sv,g=ee*qv;if(adjoint){h=conj(h);g=conj(g);}
    X[p*nn+e]+=h;Gamma[p*nn+e]+=g;
    #pragma unroll
    for(int a=0;a<3;a++){
        float2 v=zero2()-c[a]*sv;
        #pragma unroll
        for(int f=0;f<MOST_FOCUSED;f++)if(f<P.focused)v-=(pt.grad_s[f][a]*ee)*ds[f];
        G[(3*p+a)*nn+e]+=adjoint?conj(v):v;
    }
}
__device__ __forceinline__ void atomic_add(float2* x,float2 v){atomicAdd(&x->x,v.x);atomicAdd(&x->y,v.y);}

// A cluster's components at the point (model.rs `Model::components`) into X: per part q, H_q -- its
// groups' −S, at part 0 h0 less the scalar, −(i/2) its groups' Q, its explicit states' terms -- and G_q,
// then the gradients' Hermitian parts (G₀'s; per pair ±Ω, (G_Ω + G_{−Ω}†)/2 at Ω and its adjoint at −Ω).
// The lanes take the entries e = lane + 32j; a pair standing for two reads its adjoint at the mirror's place.
__device__ __noinline__ void exact_components(const Program& P,const Warp& W,const Cluster& C,int lane,float2* X){
    const int n=C.n,nn=n*n,Q=C.parts;const float2* tables=P.tables+C.table;const Point& pt=W.pt;
    float2* G=X+Q*nn;float2* Gamma=X+4*Q*nn;
    for(int e=lane;e<5*Q*nn;e+=32)X[e]=zero2();
    __syncwarp();
    for(int gi=0;gi<C.ngroups;gi++){
        const Group Gr=P.groups[C.groups+gi];if(Gr.count!=nn)continue;
        const int p=P.group_parts[C.group_parts+gi];const Fields& fa=pt.fields[Gr.absorbed];const Fields& fe=pt.fields[Gr.emitted];
        for(int pi=0;pi<Gr.npairs;pi++){
            const int2 pair=P.pairs[Gr.pairs+pi];const int k1=pair.x,k=pair.y;const bool twice=Gr.hermitian&&k1<k;
            const float2 ee=conj(fe.e[k1])*fa.e[k];float2 c[3];
            #pragma unroll
            for(int a=0;a<3;a++)c[a]=conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a];
            const float2* s=tables+Gr.s+pi*nn;const float2* q=tables+Gr.q+pi*nn;
            for(int e=lane;e<nn;e+=32){
                const int r=e/n,m=(e-r*n)*n+r;
                group_term(P,W,C,s+e,q+e,ee,c,p,false,e,X,G,Gamma,nn);
                if(twice)group_term(P,W,C,s+m,q+m,ee,c,p,true,e,X,G,Gamma,nn);
            }
        }
    }
    // The sparse groups' entries at their places (`items`), an adjoint's from its pair at the mirror.
    if(C.nbeats>0)for(int e=lane;e<nn;e+=32){
        const int end=P.item_first[C.positions+e+1];
        for(int i=P.item_first[C.positions+e];i<end;i++){
            const int4 it=__ldg(P.items+i);const int k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
            const int p=P.residual_parts[C.residual_parts+(it.z>>12)];const float2 ee=conj(fe.e[k1])*fa.e[k];float2 c[3];
            #pragma unroll
            for(int a=0;a<3;a++)c[a]=conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a];
            group_term(P,W,C,tables+it.x,tables+it.y,ee,c,p,(it.z>>11)&1,e,X,G,Gamma,nn);
        }
    }
    // h0 and its slopes; H_q = (h0 at q = 0) − M_q − (i/2)Γ_q, the static part less its scalar (its
    // Hermitian part's mean diagonal).
    float diagonal=0.f;
    for(int e=lane;e<nn;e+=32){
        float2 ds[MOST_FOCUSED];const float2 h0=entry_slopes(P,W,tables+C.h0+e,C.size,ds);
        #pragma unroll
        for(int a=0;a<3;a++){
            float2 v=zero2();
            #pragma unroll
            for(int f=0;f<MOST_FOCUSED;f++)if(f<P.focused)v+=pt.grad_s[f][a]*ds[f];
            G[a*nn+e]+=v;
        }
        const int r=e/n;if(e==r*n+r)diagonal+=h0.x-X[e].x;
        for(int p=0;p<Q;p++){const float2 m=X[p*nn+e],g=Gamma[p*nn+e];X[p*nn+e]=(p==0?h0:zero2())-m+make_float2(0.5f*g.y,-0.5f*g.x);}
    }
    const float scalar=warp_sum(diagonal)/(float)n;
    __syncwarp();
    if(lane<n)X[lane*n+lane].x-=scalar;
    // The explicit states' terms (model.rs `exact_terms`), a lane each: a kept drive's Σ_k E_k U_k, a decay
    // cross term's (i/2) Σ_k E_k X_k, a decay's value, a dressing's drive's Σ_k E_k Y_k, and their mirrors at the
    // negative residual's part; the drives' gradients.
    for(int i=lane;i<C.nterms;i+=32){
        const Term T=P.terms[C.terms+i];const int at=T.row*n+T.col,mirror=T.col*n+T.row;const Fields& fl=pt.fields[T.carrier];const int rank=P.carriers[T.carrier].rank;
        float2 v=zero2(),w=zero2(),g[3]={zero2(),zero2(),zero2()};
        if(T.kind==0){
            for(int k=0;k<rank;k++){const float2 u=make_float2(T.u[k][0],T.u[k][1]);v+=fl.e[k]*u;
                #pragma unroll
                for(int a=0;a<3;a++)g[a]+=fl.grad[k][a]*u;}
            w=conj(v);
        }else if(T.kind==1){
            float2 z=zero2();for(int k=0;k<rank;k++)z+=fl.e[k]*entry(P,W,tables+T.at+k,C.size,-1);
            v=make_float2(-0.5f*z.y,0.5f*z.x);w=make_float2(0.5f*z.y,0.5f*z.x);
        }else if(T.kind==2)v=entry(P,W,tables+T.at,C.size,-1);
        else for(int k=0;k<rank;k++){
            float2 ds[MOST_FOCUSED];const float2 y=entry_slopes(P,W,tables+T.at+k,C.size,ds);v+=fl.e[k]*y;
            #pragma unroll
            for(int a=0;a<3;a++){
                g[a]+=fl.grad[k][a]*y;
                #pragma unroll
                for(int f=0;f<MOST_FOCUSED;f++)if(f<P.focused)g[a]+=pt.grad_s[f][a]*(fl.e[k]*ds[f]);
            }
        }
        if(T.kind==3)w=conj(v);
        atomic_add(X+T.part*nn+at,v);if(T.kind!=2)atomic_add(X+T.mirror*nn+mirror,w);
        if(T.kind==0||T.kind==3){
            #pragma unroll
            for(int a=0;a<3;a++){atomic_add(G+(3*T.part+a)*nn+at,g[a]);atomic_add(G+(3*T.mirror+a)*nn+mirror,conj(g[a]));}
        }
    }
    __syncwarp();
    // The gradients' Hermitian parts: G₀'s, a pair of entries a lane's; per pair of beats ±Ω, G_Ω's entry
    // and G_{−Ω}'s at its mirror a lane's.
    for(int e=lane;e<nn;e+=32){
        const int r=e/n,c=e-r*n;if(c<r)continue;
        #pragma unroll
        for(int a=0;a<3;a++){float2* g=G+a*nn;const float2 z=0.5f*(g[e]+conj(g[c*n+r]));g[e]=z;g[c*n+r]=conj(z);}
    }
    const int half=(Q-1)/2;
    for(int i=0;i<half;i++){
        const int up=1+half+i,down=half-i;
        for(int e=lane;e<nn;e+=32){
            const int r=e/n,m=(e-r*n)*n+r;
            #pragma unroll
            for(int a=0;a<3;a++){float2* u=G+(3*up+a)*nn;float2* v=G+(3*down+a)*nn;const float2 z=0.5f*(u[e]+conj(v[m]));u[e]=z;v[m]=conj(z);}
        }
    }
    __syncwarp();
}

// The size of an n × n matrix in the scratch (engine.rs `Segment::bounds`): the larger of its largest column
// and row sums of |re| + |im|; its largest column sum (`column_norm`) into `column`.
__device__ float msize(const float2* A,int n,int lane,float& column){
    float col=0.f,row=0.f;
    if(lane<n)for(int a=0;a<n;a++){const float2 x=A[a*n+lane],y=A[lane*n+a];col+=fabsf(x.x)+fabsf(x.y);row+=fabsf(y.x)+fabsf(y.y);}
    column=warp_max(col);return fmaxf(column,warp_max(row));
}
// out = A B − B A for n × n matrices in the scratch.
__device__ void commutator(const float2* A,const float2* B,float2* out,int n,int lane){
    __syncwarp();
    for(int e=lane;e<n*n;e+=32){const int r=e/n,c=e-r*n;float2 s=zero2();for(int k=0;k<n;k++)s+=A[r*n+k]*B[k*n+c]-B[r*n+k]*A[k*n+c];out[e]=s;}
    __syncwarp();
}

// The beats a step leaves out (engine.rs `Segment::resolve`, `bounds`): per pair ±Ω its cost on the state
// and on the force's integral -- Duhamel's formula integrated by parts, its boundary terms once over the
// horizon --, the cheapest first while their sums stay within the share (`Settings::beats`). The parts
// kept (bits); their cost into `dropped`. F: per part ‖H_q‖₁, then the sizes of H_q, then of G_q per axis.
__device__ unsigned exact_resolve(const Program& P,float2* X,const float* F,const double* omegas,int n,int Q,double tau,int lane,double& dropped){
    unsigned kept=Q>=32?0xffffffffu:(1u<<Q)-1u;dropped=0.;const int half=(Q-1)/2;if(half==0)return kept;
    const int nn=n*n;float2* W1=X+(5*Q+2)*nn;float2* W2=W1+nn;
    double all=0.,pace=0.;for(int q=1;q<Q;q++){all+=(double)F[Q+q];pace+=fabs(omegas[q-1])*(double)F[Q+q];}
    const double bound=(double)F[Q]+all,once=fmin(tau/P.horizon,1.);
    double scale=0.;for(int a=0;a<3;a++)scale=fmax(scale,(double)F[2*Q+a]);scale*=tau;
    double cost[16],push[16];int order[16];
    for(int i=0;i<half;i++){
        const int down=half-i,up=1+half+i;const double w=omegas[up-1],others=pace-w*((double)F[Q+down]+(double)F[Q+up]);double x=0.;
        for(int side=0;side<2;side++){
            const int q=side?up:down;const double m=(double)F[Q+q];float unused;
            commutator(X,X+q*nn,W1,n,lane);const double x1=(double)msize(W1,n,lane,unused);
            commutator(X,W1,W2,n,lane);const double x2=(double)msize(W2,n,lane,unused);
            const double xx=x1+3.*m*all,y=x2+3.*x1*all+6.*bound*m*all+3.*m*all*all+2.*m*others+w*m*all;
            x+=fmin(2.*m/w*once+2.*bound*m*tau/w,(2.*m/w+2.*xx/(w*w))*once+tau*y/(w*w));
        }
        double f=0.;
        for(int a=0;a<3;a++){
            double v=0.;
            for(int side=0;side<2;side++){const double g=(double)F[2*Q+3*(side?up:down)+a];
                v+=fmin(2.*g/w*once+4.*bound*g*tau/w,(2.*g/w+8.*bound*g/(w*w))*once+tau*(24.*bound*bound+4.*pace)*g/(w*w));}
            f=fmax(f,v);
        }
        cost[i]=x;push[i]=f;order[i]=i;
    }
    // The cheapest first (a stable sort), while within the share.
    for(int i=1;i<half;i++){const int o=order[i];int j=i;while(j>0&&cost[order[j-1]]>cost[o]){order[j]=order[j-1];j--;}order[j]=o;}
    double moved=0.,pushed=0.;
    for(int j=0;j<half;j++){
        const int i=order[j];
        if(moved+cost[i]<=P.beats&&pushed+push[i]<=P.beats*scale){moved+=cost[i];pushed+=push[i];kept&=~((1u<<(half-i))|(1u<<(1+half+i)));}else break;
    }
    dropped=moved;return kept;
}

// A cluster's segment of an exact step of τ from `start` at the point (engine.rs `Segment::new`): its
// components, their norms and phases at the start, the beats it leaves out (their bound per unit time
// into `beats`, the trajectory's largest), and its grid: with beats kept, the spacing whose phase
// (‖H₀‖₁ + Σ‖H_j‖₁ + max|Ω|/2)δ is within the node phase -- with a period, its k-th part where the
// CPU caches the period's propagators --, else ‖H₀‖₁δ within it and its decay i(H₀ − H₀†) for the roots.
__device__ __noinline__ Segment exact_segment(const Program& P,const Warp& W,int k,double start,double tau,int lane,float2* X,float& beats){
    const Cluster C=P.clusters[k];Segment S;S.n=C.n;S.parts=C.parts;S.start=start;
    const int n=C.n,nn=n*n,Q=C.parts;const double* omegas=P.omegas+C.omegas;
    float2* base=X+(5*Q+4)*nn;float* F=(float*)(base+Q);float* w=F+5*Q;S.base=base;S.F=F;S.w=w;
    exact_components(P,W,C,lane,X);
    for(int m=0;m<4*Q;m++){float column;const float s=msize(X+m*nn,n,lane,column);if(m<Q){F[m]=column;F[Q+m]=s;}else F[2*Q+m-Q]=s;}
    if(lane<Q){base[lane]=lane==0?make_float2(1.f,0.f):cis(omegas[lane-1]*start);w[lane]=lane==0?0.f:(float)omegas[lane-1];}
    __syncwarp();
    double dropped;S.kept=exact_resolve(P,X,F,omegas,n,Q,tau,lane,dropped);beats=fmaxf(beats,(float)(dropped/tau));
    S.beating=(S.kept&~1u)!=0u;
    if(!S.beating){
        const int steps=max((int)ceil((double)F[0]*tau/P.node_phase),1);S.spacing=tau/(double)steps;
        float2* D=X+(5*Q+1)*nn;
        for(int e=lane;e<nn;e+=32){const int r=e/n,c=e-r*n;const float2 z=X[e]-conj(X[c*n+r]);D[e]=make_float2(-z.y,z.x);}
        __syncwarp();
    }else{
        double norm=(double)F[0],most=0.;
        for(int q=1;q<Q;q++)if((S.kept>>q)&1u){norm+=(double)F[q];most=fmax(most,fabs(omegas[q-1]));}
        const double widest=P.node_phase/(norm+0.5*most),period=omegas[Q-1];bool cached=false;
        if(period>0.){const int kk=max((int)ceil(period/widest),1);const double sp=period/(double)kk;if(tau/sp>(double)(kk*n)){S.spacing=sp;cached=true;}}
        if(!cached)S.spacing=tau/(double)max((int)ceil(tau/widest),1);
    }
    S.delta=(float)S.spacing;
    return S;
}

// The root s ∈ (0, width] of ln‖x(s)‖² = target from the panel's start x and middle (engine.rs
// `Segment::root`): Newton on the logarithm with its derivative −rate/‖x‖², from linear interpolation on
// the half it lies in, kept in the bracket (bisection where it steps out), to single precision.
__device__ float exact_root(const Segment& S,const float2* X,float2 x,float2 middle,float t,float na,float nm,float nb,float width,float target,int lane){
    const float half=0.5f*width,lna=logf(na),lnm=logf(nm),lnb=logf(nb);
    float lo,hi,la,lb;if(lnm<=target){lo=0.f;hi=half;la=lna;lb=lnm;}else{lo=half;hi=width;la=lnm;lb=lnb;}
    float s=lb<la?fminf(fmaxf(lo+(target-la)/(lb-la)*(hi-lo),lo),hi):hi;
    for(int it=0;it<64;it++){
        const bool first=s<=half;const float from=first?0.f:half;
        const float2 y=seg_advance(S,X,t+from,s-from,false,first?x:middle,lane);
        const float norm=vnorm(y,lane),g=logf(norm)-target;
        if(g>0.f)lo=s;else hi=s;
        const float rate=seg_rate(S,X,t+s,y,lane)/norm;
        float next=rate>0.f?s+g/rate:0.5f*(lo+hi);
        if(!(next>lo&&next<hi))next=0.5f*(lo+hi);
        if(fabsf(next-s)<=1e-6f*fmaxf(width,s))return next;
        s=next;
    }
    return s;
}

// From x at `at` (single precision from the step's start; its ln‖x‖² relative to the trajectory's), the
// evolution for at most `remaining` until ln‖x‖² falls to `target` (ln θ less the trajectory's ln‖ψ‖²,
// engine.rs `Segment::evolve`): composite Simpson on panels of two spacings from a node -- a panel to the
// next node first where it starts between nodes (after a jump), what is left last --, the impulse ∫⟨F⟩dt
// into `impulse` (lab); the time taken, and `jumped` where it reached it (x then at the root).
__device__ float exact_evolve(const Segment& S,const float2* X,float2& x,float at,float remaining,float target,int lane,bool& jumped,float impulse[3]){
    const float end=at+remaining,delta=S.delta;float t=at;
    int i=max((int)ceilf(at/delta),0);
    float fa[3];float na=seg_force(S,X,t,x,lane,fa);jumped=false;
    while(t<end){
        const float node=(float)i*delta;float b;bool whole=false;
        if(node>t)b=fminf(node,end);else{const float b2=(float)(i+2)*delta;if(b2<=end){b=b2;whole=true;}else b=end;}
        const float width=b-t,half=0.5f*width;
        float2 middle,last;seg_advances(S,X,t,half,whole,x,lane,middle,last);
        float fm[3],fb[3],nm,nb;seg_forces(S,X,t+half,b,middle,last,lane,fm,fb,nm,nb);
        if(logf(nb)<=target){
            const float s=exact_root(S,X,x,middle,t,na,nm,nb,width,target,lane);
            const float2 h=seg_advance(S,X,t,0.5f*s,false,x,lane),y=seg_advance(S,X,t+0.5f*s,0.5f*s,false,h,lane);
            float fh[3],fs[3];seg_force(S,X,t+0.5f*s,h,lane,fh);seg_force(S,X,t+s,y,lane,fs);
            #pragma unroll
            for(int a=0;a<3;a++)impulse[a]+=s/6.f*(fa[a]+4.f*fh[a]+fs[a]);
            x=y;jumped=true;return t+s-at;
        }
        #pragma unroll
        for(int a=0;a<3;a++){impulse[a]+=width/6.f*(fa[a]+4.f*fm[a]+fb[a]);fa[a]=fb[a];}
        x=last;na=nb;t=b;if(whole)i+=2;
    }
    return remaining;
}

// The exact step of τ (engine.rs `Engine::exact`): the flow for τ/2 at the step's weight w (`weight`, at its start); at the point there every cluster the
// trajectory visits takes its segment once (`exact_segment`), ψ through the step with the jumps at their
// roots and the impulse along it; the kick, less τ w times the harmonic force at the point; the flow for τ/2 at w.
__device__ __noinline__ void exact_step(const Program& P,const Run& R,Warp& W,Traj& t,double tau,bool exact,int lane,float2* scratch){
    COUNT(COUNT_ADVANCES);
    const double w=weight(P,t.q);
    flow(P,t,0.5*tau,w);point(P,W,t,lane);
    const double start=t.time,ph[3]={t.p[0],t.p[1],t.p[2]};float2* X=scratch+P.exact_at;
    double impulse[3]={0.,0.,0.},elapsed=0.;int built=-1;Segment S;
    __syncwarp();float2 x=W.psi[lane&15];
    while(elapsed<tau){
        if(built!=t.cluster){
            // A later cluster's segment at the step's point (the jump's moved W.pt to its own momentum).
            if(built>=0){Traj h=t;for(int i=0;i<3;i++)h.p[i]=ph[i];point(P,W,h,lane);}
            SPAN_BEGIN(segment);S=exact_segment(P,W,t.cluster,start,tau,lane,X,t.beats);S.v=W.vec;built=t.cluster;SPAN_END(SPAN_GENERATOR,segment);
        }
        bool jumped;float part[3]={0.f,0.f,0.f};
        SPAN_BEGIN(evolving);const float taken=exact_evolve(S,X,x,(float)elapsed,(float)(tau-elapsed),(float)(t.log_threshold-t.logn),lane,jumped,part);SPAN_END(SPAN_EVOLVE,evolving);
        for(int a=0;a<3;a++)impulse[a]+=(double)part[a];
        elapsed=jumped?elapsed+(double)taken:tau;
        if(!jumped)break;
        __syncwarp();if(lane<16)W.psi[lane]=x;__syncwarp();
        t.time=start+elapsed;
        {COUNT(COUNT_JUMPS);SPAN_BEGIN(jumping);jump(P,R,W,t,exact,lane,scratch);SPAN_END(SPAN_JUMP,jumping);}
        __syncwarp();x=W.psi[lane&15];
    }
    const float norm=vnorm(x,lane);t.logn+=log((double)norm);
    __syncwarp();if(lane<16)W.psi[lane]=(1.f/sqrtf(norm))*x;__syncwarp();
    #pragma unroll
    for(int i=0;i<3;i++)t.p[i]+=P.axes[i][0]*impulse[0]+P.axes[i][1]*impulse[1]+P.axes[i][2]*impulse[2]+tau*w*P.stiffness[i]*t.q[i];
    flow(P,t,0.5*tau,w);t.time=start+tau;
}

// The trap's potential of ψ (W.psi) at the point for a cluster with explicit states (model.rs
// `trap_mixed`): Re⟨ψ|(h0 − M) here − (h0 − M) with the trap's light off|ψ⟩/⟨ψ|ψ⟩, the groups' pairs
// standing for two taken twice; with `bound`, h0's share and twice the light shift's bound.
__device__ float trap_exact(const Program& P,const Warp& W,const Cluster& C,double t,bool bound,int lane){
    const int n=C.n,nn=n*n;const float2* tables=P.tables+C.table;const Point& pt=W.pt;float s=0.f;
    for(int e=lane;e<nn;e+=32){
        const int r=e/n,c=e-r*n;float2 d=entry(P,W,tables+C.h0+e,C.size,-1)-far_entry(P,W,tables+C.h0+e,C.size);
        if(!bound){
            for(int gi=0;gi<C.ngroups;gi++){
                const Group G=P.groups[C.groups+gi];if(G.count!=nn)continue;
                const Fields& fa=pt.fields[G.absorbed];const Fields& fe=pt.fields[G.emitted];const float2 phase=residual(G.residual,t);
                for(int pi=0;pi<G.npairs;pi++){
                    const int2 pair=P.pairs[G.pairs+pi];const int k1=pair.x,k=pair.y;const float both=G.hermitian&&k1<k?2.f:1.f;
                    const float2 cc=zero2()-both*(conj(fe.e[k1])*fa.e[k])*phase,c0=both*(conj(pt.off[G.emitted][k1])*pt.off[G.absorbed][k])*phase;
                    const float2* x=tables+G.s+pi*nn+e;d+=cc*entry(P,W,x,C.size,-1)+c0*far_entry(P,W,x,C.size);
                }
            }
            if(C.nbeats>0){
                const int end=P.item_first[C.positions+e+1];
                for(int i=P.item_first[C.positions+e];i<end;i++){
                    const int4 it=__ldg(P.items+i);if((it.z>>11)&1)continue;
                    const int k1=it.z&3,k=(it.z>>2)&3,ca=(it.z>>4)&7,ce=(it.z>>7)&7;const float both=(it.z>>10)&1?2.f:1.f;
                    const float2 phase=residual(P.residuals[C.beats+(it.z>>12)],t);
                    const float2 cc=zero2()-both*(conj(pt.fields[ce].e[k1])*pt.fields[ca].e[k])*phase,c0=both*(conj(pt.off[ce][k1])*pt.off[ca][k])*phase;
                    const float2* x=tables+it.x;d+=cc*entry(P,W,x,C.size,-1)+c0*far_entry(P,W,x,C.size);
                }
            }
        }
        const float2 z=d*W.psi[c];s+=W.psi[r].x*z.x+W.psi[r].y*z.y;
    }
    float norm=0.f;for(int b=0;b<n;b++)norm+=norm2(W.psi[b]);
    const float v=warp_sum(s)/norm;
    return bound?v+P.light2:v;
}

// The photon's recoils (engine.rs `kick`): −k n, the absorbed beam drawn by its share (`absorbed`, the
// shares in J.shares and J.beams, `count` of them) less their mean, and the exact conditional mean
// (`mean`, its Im⟨f|∇f⟩ over ‖f‖²) -- onto the momentum on the trap's principal axes.
__device__ void recoil(const Program& P,const JumpSpace& J,Traj& t,const double n[3],double wave_number,const double mean[3],int count,double absorbed){
    double kick[3],sum=0.;for(int a=0;a<3;a++)kick[a]=-wave_number*n[a];
    for(int i=0;i<count;i++)sum+=(double)J.shares[i];
    if(sum>0.){
        double x=absorbed*sum;int drawn=J.beams[count-1];
        for(int i=0;i<count;i++){const double w=(double)J.shares[i];if(x<w){drawn=J.beams[i];break;}x-=w;}
        double average[3]={0.,0.,0.};
        for(int i=0;i<count;i++){const Beam& b=P.beams[J.beams[i]];for(int a=0;a<3;a++)average[a]+=(double)J.shares[i]*((double)b.direction[a]*(double)b.k);}
        const Beam& b=P.beams[drawn];
        for(int a=0;a<3;a++)kick[a]+=(double)b.direction[a]*(double)b.k+mean[a]-average[a]/sum;
    }else for(int a=0;a<3;a++)kick[a]+=mean[a];
    #pragma unroll
    for(int i=0;i<3;i++)t.p[i]+=P.axes[i][0]*kick[0]+P.axes[i][1]*kick[1]+P.axes[i][2]*kick[2];
}

// A table entry of a carrier's response T_k at the stencil (model.rs `responses`' tables: rank × NE × n0).
__device__ __forceinline__ const float2* t_table(const Program& P,const Cluster& C,int c,int k,int i,int col){
    return P.tables+C.table+P.carrier_tables[C.carriers+3*c+2]+(k*NE+i)*C.n0+col;
}

// The post-jump transient into a cluster with explicit states (engine.rs `transient`; model.rs `transient`,
// `transient_photon`) from the normalized ground state ψ′ the jump left (W.psi, its n0 ground states) with
// the jump's draws u: per coherent group k its folded amplitude e_k = Σ e^{−iε̃t} R_c ψ′ (X_k ψ′ = −e_k), p =
// Σ_k ‖e_k‖²; with probability 1 − p ψ′ ← (1 − ½Y)ψ′, Yψ′ = Σ_k X_k† X_k ψ′; else one more photon from a piece
// (k, excited group, target cluster), S_a = P_B C_a D^{−1/2} X_k on the group's levels, drawn by ‖S ψ′‖²,
// its kicks a jump's from transient_photon's amplitude (Σ e^{−iε̃t} T_k ψ′, the opposite sign), its
// gradient and its beams' parts. E: the scratch's room for e_k.
__device__ __noinline__ void transient(const Program& P,const Run& R,Warp& W,Traj& t,const double u[4],Rng& rng,int lane,float2* E){
    const Cluster C=P.clusters[t.cluster];const int n0=C.n0;const Point& pt=W.pt;const double time=t.time;JumpSpace& J=W.jump;
    // e_k for every coherent group, lane i its row i.
    float p=0.f;
    for(int k=0;k<C.ncoherent;k++){
        const Coherent G=P.coherent[C.coherent+k];float2 e=zero2();
        if(lane<NE)for(int h=0;h<G.count;h++){const Channel ch=P.channels[G.channels+h];e+=residual(-ch.residual,time)*response(P,W,C,ch.carrier,lane,ch.col)*W.psi[ch.col];}
        if(lane<NE)E[k*NE+lane]=e;p+=lane<NE?norm2(e):0.f;
    }
    p=warp_sum(p);__syncwarp();
    if(!(p>0.f))return;
    if(u[0]>=(double)p){
        // No photon: ψ′ ← (1 − ½Y)ψ′, (Yψ′)_b = Σ_k Σ_{channels at b} e^{iε̃t} Σ_i R_c[i,b]* e_k[i].
        float2 y=zero2();
        for(int k=0;k<C.ncoherent;k++){
            const Coherent G=P.coherent[C.coherent+k];
            for(int h=0;h<G.count;h++){
                const Channel ch=P.channels[G.channels+h];
                const float2 v=lane<NE?conj(response(P,W,C,ch.carrier,lane,ch.col))*E[k*NE+lane]:zero2();
                const float2 s=residual(ch.residual,time)*make_float2(warp_sum(v.x),warp_sum(v.y));
                if(lane==ch.col)y+=s;
            }
        }
        float2 next=lane<n0?W.psi[lane]-0.5f*y:zero2();const float norm=warp_sum(lane<n0?norm2(next):0.f);
        __syncwarp();if(lane<n0)W.psi[lane]=(1.f/sqrtf(norm))*next;__syncwarp();
        return;
    }
    // The pieces in order (k, excited group, target): their amplitudes' weights, the drawn one by u[1].
    // Lane 3·to + a takes the photon's axis a at ground state `to`: s[to, b] ψ′_b = Σ_{i ∈ group} C_a[to, i]
    // D_ii^{−1/2} (X_k ψ′)_i; a target is a cluster of the ground states its operator reaches.
    const int to=lane/3,ax=lane-3*to;double total=0.,x=0.;int drawn[3]={-1,-1,-1},last[3]={-1,-1,-1};
    for(int pass=0;pass<2;pass++){
        if(pass==1){if(!(total>0.))return;x=u[1]*total;}
        for(int k=0;k<C.ncoherent;k++){
            const Coherent G=P.coherent[C.coherent+k];
            for(int g=0;g<P.nlevels;g++){
                const int first=P.level_first[g],end=P.level_first[g+1];
                // The group's X on its levels: none where every channel's response there vanishes.
                bool any=false;
                for(int h=0;h<G.count&&!any;h++){const Channel ch=P.channels[G.channels+h];for(int j=first;j<end;j++)if(!zero(response(P,W,C,ch.carrier,P.levels[j],ch.col))){any=true;break;}}
                if(!any)continue;
                // The operator's rows reached, and the amplitude at this lane's (to, axis).
                bool reached=false;float2 v=zero2();
                if(lane<3*NG)for(int j=first;j<end;j++){
                    const int i=P.levels[j];const float2 c=P.emission[(ax*NG+to)*NE+i];if(zero(c))continue;
                    for(int h=0;h<G.count;h++){const Channel ch=P.channels[G.channels+h];if(!zero(response(P,W,C,ch.carrier,i,ch.col)))reached=true;}
                    v+=(-P.decay_root[i])*c*E[k*NE+i];
                }
                const unsigned rows=__ballot_sync(FULL,reached);
                // Each target cluster in order: its weight Σ ‖v‖² over its ground states.
                for(int target=0;target<P.nclusters;target++){
                    bool hit=false;for(int b=0;b<NG;b++)if(P.ground_cluster[b]==target&&((rows>>(3*b))&7u))hit=true;
                    if(!hit)continue;
                    const double w=(double)warp_sum(lane<3*NG&&P.ground_cluster[to]==target?norm2(v):0.f);
                    if(pass==0){total+=w;continue;}
                    if(drawn[0]>=0)continue;
                    last[0]=k;last[1]=g;last[2]=target;
                    if(x<w){drawn[0]=k;drawn[1]=g;drawn[2]=target;}else x-=w;
                }
            }
        }
    }
    if(drawn[0]<0){drawn[0]=last[0];drawn[1]=last[1];drawn[2]=last[2];}
    if(drawn[0]<0)return;
    // The drawn piece's amplitude u_a on the target's rows into J.u: its photon f, direction and polarization.
    const int k=drawn[0],g=drawn[1],target=drawn[2];const Cluster T=P.clusters[target];const int nt=T.n0;
    const Coherent G=P.coherent[C.coherent+k];const int first=P.level_first[g],stop=P.level_first[g+1];
    {
        float2 v=zero2();
        if(lane<3*NG&&P.ground_cluster[to]==target)for(int j=first;j<stop;j++){const int i=P.levels[j];v+=(-P.decay_root[i])*P.emission[(ax*NG+to)*NE+i]*E[k*NE+i];}
        __syncwarp();
        if(lane<3*MAXN){const int row=lane/3;J.u[lane%3][row]=zero2();}
        __syncwarp();
        if(lane<3*NG&&P.ground_cluster[to]==target)J.u[ax][P.ground_position[to]]=v;
        __syncwarp();
    }
    double n[3],e[3];const float size=emit(rng,J,nt,u[2],lane,n,e);
    // transient_photon: the gradient along e of its amplitude Σ_{i} C_a D^{−1/2} Σ_channels e^{−iε̃t} Σ_k
    // (∇E_k T_k + E_k Σ_f ∂_f T_k ∇s_f) ψ′ per row, and per beam of the group's carriers its part
    // a_l Σ_k c_lk T_k: the exact conditional mean and the beams' shares.
    float mean[3]={0.f,0.f,0.f};
    if(lane<NG&&P.ground_cluster[lane]==target){
        const int row=P.ground_position[lane];
        for(int j=first;j<stop;j++){
            const int i=P.levels[j];float2 c=zero2();
            for(int a=0;a<3;a++)c+=(float)e[a]*P.emission[(a*NG+lane)*NE+i];
            c=P.decay_root[i]*c;if(zero(c))continue;
            for(int h=0;h<G.count;h++){
                const Channel ch=P.channels[G.channels+h];const Fields& fl=pt.fields[ch.carrier];const float2 phase=residual(-ch.residual,time)*W.psi[ch.col];
                for(int kk=0;kk<P.carriers[ch.carrier].rank;kk++){
                    float2 ds[MOST_FOCUSED];const float2 tv=entry_slopes(P,W,t_table(P,C,ch.carrier,kk,i,ch.col),C.size,ds);
                    #pragma unroll
                    for(int x=0;x<3;x++){
                        float2 d=fl.grad[kk][x]*tv;
                        #pragma unroll
                        for(int f=0;f<MOST_FOCUSED;f++)if(f<P.focused)d+=pt.grad_s[f][x]*(fl.e[kk]*ds[f]);
                        const float2 z=c*phase*d,fr=J.f[row];mean[x]+=fr.x*z.y-fr.y*z.x;
                    }
                }
            }
        }
    }
    double m[3];for(int x=0;x<3;x++)m[x]=(double)warp_sum(mean[x])/(double)size;
    // The beams' shares: the group's carriers ascending, each one's beams in order.
    int count=0;
    for(int c=0;c<P.ncarriers;c++){
        bool in=false;for(int h=0;h<G.count;h++)if(P.channels[G.channels+h].carrier==c)in=true;
        if(!in)continue;const Carrier Rc=P.carriers[c];
        for(int b=0;b<Rc.beams;b++){
            const int l=P.beam_list[Rc.first+b];const Beam& B=P.beams[l];float2 v=zero2();
            if(lane<NG&&P.ground_cluster[lane]==target){
                for(int j=first;j<stop;j++){
                    const int i=P.levels[j];float2 ce=zero2();
                    for(int a=0;a<3;a++)ce+=(float)e[a]*P.emission[(a*NG+lane)*NE+i];
                    ce=P.decay_root[i]*ce;if(zero(ce))continue;
                    for(int h=0;h<G.count;h++){
                        const Channel ch=P.channels[G.channels+h];if(ch.carrier!=c)continue;
                        float2 z=zero2();for(int kk=0;kk<Rc.rank;kk++)z+=make_float2(B.coefficients[kk][0],B.coefficients[kk][1])*entry(P,W,t_table(P,C,c,kk,i,ch.col),C.size,-1);
                        v+=ce*residual(-ch.residual,time)*W.psi[ch.col]*pt.a[l]*z;
                    }
                }
            }
            const float w=warp_sum(norm2(v));if(lane==0){J.shares[count]=w;J.beams[count]=l;}count++;
        }
    }
    __syncwarp();
    // The photon's wave number: its group's first channel's carrier's (model.rs `Piece::wave_number`).
    recoil(P,J,t,n,P.carriers[P.channels[G.channels].carrier].wave,m,count,u[3]);
    const float scale=1.f/sqrtf(size);
    __syncwarp();if(lane<MAXM)W.psi[lane]=lane<nt?scale*J.f[lane]:zero2();__syncwarp();
    t.cluster=target;
}

// One step of dt (engine.rs `Engine::step`): flow, kick and flow; where ‖ψ‖² crosses the threshold
// the step is redone to the crossing, the jump made there and the rest taken from the state it left.
// True when the atom is lost in it: unbound -- its mechanical energy above the trap's far level --
// outside the trap's region, judged at the step's middle where it went without a jump, else at its
// end.
__device__ bool step(const Program& P,const Run& R,Warp& W,Traj& t,int lane,float2* scratch){
    const float ratio=doppler_ratio(P,t,lane);const bool exact=ratio>P.allowance;
    if(!exact)t.doppler=fmaxf(t.doppler,ratio);
    double left=P.dt;Local L;Middle M;bool fresh=false;
    if(P.clusters[t.cluster].exact){exact_step(P,R,W,t,left,exact,lane,scratch);point(P,W,t,lane);fresh=true;}
    else for(;;){
        const Traj start=t;
        if(lane<MAXN)W.start[lane]=W.psi[lane];
        double lh;advance(P,W,t,left,lane,lh,L,M);
        if(t.logn>t.log_threshold)break;
        // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
        double ta,tb,la,lb;
        if(lh<=t.log_threshold){ta=0.;tb=0.5*left;la=start.logn;lb=lh;}else{ta=0.5*left;tb=left;la=lh;lb=t.logn;}
        const double at=lb<la?fmin(fmax(ta+(t.log_threshold-la)/(lb-la)*(tb-ta),ta),tb):tb;
        t=start;__syncwarp();if(lane<MAXN)W.psi[lane]=W.start[lane];__syncwarp();
        if(at>0.){double unused;advance(P,W,t,at,lane,unused,L,M);}
        {COUNT(COUNT_JUMPS);SPAN_BEGIN(jumping);jump(P,R,W,t,exact,lane,scratch);SPAN_END(SPAN_JUMP,jumping);}
        left-=at;
        if(!(left>0.)){point(P,W,t,lane);fresh=true;break;}
        if(P.clusters[t.cluster].exact){exact_step(P,R,W,t,left,exact,lane,scratch);point(P,W,t,lane);fresh=true;break;}
    }
    if(!P.loss)return false;
    SPAN_BEGIN(loss);
    // The state judged: the middle's (W.pt and L.h0 are its point's) where the step went without a jump.
    const double* q=fresh?t.q:M.q;const double* p=fresh?t.p:M.p;const double time=fresh?t.time:t.time-0.5*left;
    double harmonic=0.;
    #pragma unroll
    for(int i=0;i<3;i++)harmonic+=P.omega[i]*P.omega[i]*q[i]*q[i];
    bool lost=false;
    if(0.5*harmonic/P.hm64>P.region){
        const double kinetic=0.5*P.hm64*(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
        const Cluster C=P.clusters[t.cluster];
        if(C.n>C.n0){
            // A cluster with explicit states (after the exact step, its end).
            const double kinetic=0.5*P.hm64*(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
            lost=kinetic+(double)trap_exact(P,W,C,time,true,lane)>0.&&kinetic+(double)trap_exact(P,W,C,time,false,lane)>0.;
            SPAN_END(SPAN_LOSS,loss);return lost;
        }
        const Entries E(C.n,lane);
        if(fresh){float2 slope[MOST_FOCUSED][SLOTS];gather<false>(P,W,false,P.tables+C.table+C.h0,C.size,E.nn,lane,L.h0,slope);}
        // ψ at the middle in W.psi while judged, the end's kept.
        float2 end=zero2();
        if(!fresh){__syncwarp();if(lane<MAXN){end=W.psi[lane];W.psi[lane]=M.psi;}__syncwarp();}
        lost=kinetic+(double)trap(P,W,C,E,L.h0,time,true,lane)>0.&&kinetic+(double)trap(P,W,C,E,L.h0,time,false,lane)>0.;
        if(!fresh){__syncwarp();if(lane<MAXN)W.psi[lane]=end;__syncwarp();}
    }
    SPAN_END(SPAN_LOSS,loss);
    return lost;
}

// ---------------------------------------------------------------------------------------------
// The record of an output.

// R_c[i][col] = Σ_k E_k T_k[i][col] at the stencil (model.rs `populations`' responses).
__device__ float2 response(const Program& P,const Warp& W,const Cluster& C,int c,int i,int col){
    const float2* x=P.tables+C.table+P.carrier_tables[C.carriers+3*c+2]+i*C.n0+col;const Fields& f=W.pt.fields[c];float2 r=zero2();
    for(int k=0;k<P.carriers[c].rank;k++)r+=f.e[k]*entry(P,W,x+k*NE*C.n0,C.size,-1);
    return r;
}

// The held atom's record of an output (stats.rs `Group::add`): held, the profile error, the
// (kv/σ)², the 32 populations (model.rs `populations`: the bare ground ones ρ − ½{Y, ρ}, the
// excited ones Σ |Σ_coherent e^{−iε̃t} (R ψ)_i|²) at its point, into `out` (zeros there before).
__device__ void record(const Program& P,Warp& W,const Traj& t,int lane,double* out){
    point(P,W,t,lane);
    const Cluster C=P.clusters[t.cluster];const int n=C.n;
    float profile=0.f,ratio=0.f;
    if(lane<P.nunits){
        const Unit U=P.units[lane];
        if(U.profile>=0.f){float2 F=zero2();for(int j=0;j<U.count;j++)F+=W.pt.a[P.unit_beams[U.first+j]]*P.unit_weights[U.first+j];profile=fabsf(norm2(F)/U.norm-U.focus)*U.profile;}
    }
    if(lane<P.nbeams){
        const Beam& b=P.beams[lane];
        if(b.inv_pole>0.f){const float x=W.pt.shift[lane]*b.inv_pole;ratio=x*x;}
    }
    profile=warp_max(profile);ratio=warp_max(ratio);
    // In double precision from the responses and ψ as they are: Re⟨ψ|Y|ψ⟩ is then Σ_i of the excited
    // ones to rounding, and the populations sum to one.
    double norm=0.;for(int b=0;b<n;b++)norm+=dnorm2(d2(W.psi[b]));
    double2* y=W.readout;
    if(lane<MAXN)y[lane]=make_double2(0.,0.);
    __syncwarp();
    // Each channel's response in its excited row, once: the first CACHED of a group in the matrices' space.
    constexpr int CACHED=3*MAXN*MAXN/NE;float2* cache=W.m[0];
    double excited=0.;
    for(int gi=0;gi<C.ncoherent;gi++){
        const Coherent G=P.coherent[C.coherent+gi];double2 e=make_double2(0.,0.);
        if(lane<NE)for(int h=0;h<G.count;h++){
            const Channel ch=P.channels[G.channels+h];const float2 r=response(P,W,C,ch.carrier,lane,ch.col);if(h<CACHED)cache[h*NE+lane]=r;
            const double2 z=dmul(dmul(d2(residual(-ch.residual,t.time)),d2(r)),d2(W.psi[ch.col]));e.x+=z.x;e.y+=z.y;
        }
        for(int h=0;h<G.count;h++){
            const Channel ch=P.channels[G.channels+h];double2 z=make_double2(0.,0.);
            if(lane<NE)z=dmul(dmul(d2(residual(ch.residual,t.time)),d2(conj(h<CACHED?cache[h*NE+lane]:response(P,W,C,ch.carrier,lane,ch.col)))),e);
            z=make_double2(warp_sum(z.x),warp_sum(z.y));if(lane==0){y[ch.col].x+=z.x;y[ch.col].y+=z.y;}
            __syncwarp();
        }
        excited+=dnorm2(e)/norm;
    }
    const double folded=warp_sum(lane<NE?excited:0.);
    __syncwarp();
    if(lane<3)out[lane]=lane==0?1.:lane==1?(double)profile:(double)ratio;
    if(lane==0)out[FOLDED]=folded;
    // An explicit state's population |ψ_e|²/‖ψ‖² onto its excited state's.
    if(lane<NE)for(int i=0;i<n-C.n0;i++)if(P.excited_list[C.excited+i]==lane)excited+=dnorm2(d2(W.psi[C.n0+i]))/norm;
    if(lane<NE)out[3+P.excited_states[lane]]=excited;
    if(lane<NG){
        const int x=lane<C.n0?lane:-1;double v=0.;
        if(x>=0){const double2 p=d2(W.psi[x]),q=y[x];v=(dnorm2(p)-(p.x*q.x+p.y*q.y))/norm;}
        // A ground state outside the cluster has no population; the cluster's own by its position.
        if(x>=0)out[3+P.ground_states[C.states[x]]]=v;
    }
    __syncwarp();
}

// ---------------------------------------------------------------------------------------------
// The motion windows (lib.rs `Windows`): output o's window along an axis is the steps within
// min(h, o·steps) of o·steps, h the axis's half-width, so it ends at c(o) = o·steps + min(h, o·steps),
// which grows with o: in an interval at most one whole window and one early one end. A warp keeps the
// prefix sums of its steps' (1, κ, κ², r, r²) per lab axis -- lane 5a + v holds axis a's v -- and a
// ring of them per axis (a power of two of at least 2h + steps + 1 steps), long enough to hold a
// window's start until the interval its end falls in.
__device__ __forceinline__ int window_end(const Run& R,int a,int o){const int at=o*R.steps;return at+min(R.half[a],at);}
__device__ __forceinline__ int ring_at(const Run& R,int a,int k){return R.ring_at[a]+5*(k&(R.ring[a]-1));}

// ---------------------------------------------------------------------------------------------
// The kernel: every task of the wave (one block each: its first trajectory, its count, its first
// slot) over the output intervals [begin, end). Interval 0 is the initial state; interval i > 0
// takes the steps ((i − 1)·steps, i·steps], the last ones past the last output its reach. Each warp
// runs its trajectory through the launch's intervals on its own, keeping its record of each (the
// output's point record, the windows closing in it) in its slot's records; at the launch's end the
// block sums them per interval in the trajectories' order -- the CPU's order within a task -- into
// the task's sums.
extern "C" __global__ void __launch_bounds__(WARPS*32,2) motion3d(const Program P,const Run R){
    __shared__ Warp warps[WARPS];
    const int lane=threadIdx.x&31,wi=threadIdx.x>>5;Warp& W=warps[wi];
    const int count=R.tasks[3*blockIdx.x+1],first=R.tasks[3*blockIdx.x+2],slot=first+wi;const bool active=wi<count;
    State* S=R.states+slot;double* ring=R.rings+(size_t)slot*R.ring_stride;float2* scratch=R.scratch+(size_t)slot*R.scratch_stride;
    double* records=R.records+(size_t)slot*MOST_INTERVALS*SLOT_RECORD;
    Traj t;Twin h;double prefix=0.,q0[3],p0[3];const bool held=R.held.u!=nullptr;
    long long twin_cycles=0;const long long launched=clock64();
    if(active){
        #pragma unroll
        for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];q0[i]=S->q0[i];p0[i]=S->p0[i];}
        t.time=S->time;t.logn=S->logn;t.threshold=S->threshold;t.log_threshold=S->log_threshold;
        t.doppler=S->doppler;t.cluster=S->cluster;t.jumps=S->jumps;t.death=S->death;t.index=S->index;t.beats=S->beats;
        h.logn=S->twin_logn;h.threshold=S->twin_threshold;h.log_threshold=S->twin_log_threshold;h.cluster=S->twin_cluster;h.jumps=S->twin_jumps;
        if(lane<15)prefix=S->prefix[lane];
        if(lane<MAXM)W.psi[lane]=S->psi[lane];
        if(lane<MAXN)W.twin[lane]=S->twin[lane];
    }
#ifdef PROFILE
    if(lane<SPANS)W.profile[lane]=0;
#endif
    __syncwarp();SPAN_BEGIN(launch);
    const int last=(R.outputs-1)*R.steps+R.reach;
    // Per lab axis the first output whose window has not ended before this launch.
    int start[3];
    #pragma unroll
    for(int a=0;a<3;a++){const int before=R.begin==0?-1:(R.begin-1)*R.steps;int o=max(0,floordiv(before-R.half[a],R.steps));while(window_end(R,a,o)<=before)o++;start[a]=o;}
    if(active){
        int next[3]={start[0],start[1],start[2]};
        for(int i=R.begin;i<R.end;i++){
            const int lo=i==0?-1:(i-1)*R.steps,hi=i==0?0:min(i*R.steps,last);
            // The interval's steps, their windows' sums (while held) and rings.
            for(int k=lo+1;k<=hi;k++){
                if(t.death<0){
                    SPAN_BEGIN(stepping);
                    if(k>0&&step(P,R,W,t,lane,scratch))t.death=k;
                    SPAN_END(SPAN_STEP,stepping);
                    if(t.death<0&&lane<15){
                        const int a=lane/5,v=lane-5*a;float r[3],kappa[3];lab32(P,t.q,r);lab32(P,t.p,kappa);
                        const double x=(double)(v<3?kappa[a]:r[a]);prefix+=v==0?1.:v==1||v==3?x:x*x;
                    }
                }
                if(lane<15)ring[ring_at(R,lane/5,k)+lane%5]=prefix;
            }
            // The held twin's steps of the interval, up to the last output.
            if(held&&i>0&&i<R.outputs){
                const long long begun=clock64();
                for(int k=lo+1;k<=hi;k++)hold_step(P,R,W,h,t.index,lane);
                twin_cycles+=clock64()-begun;
            }
            // The record of output i.
            double* out=records+(size_t)(i-R.begin)*SLOT_RECORD;
            for(int x=lane;x<SLOT_RECORD;x+=32)out[x]=0.;
            __syncwarp();
            SPAN_BEGIN(readout);
            if(i<R.outputs&&t.death<0){
                record(P,W,t,lane,out);
                __syncwarp();
                for(int k=0;k<R.nlisted;k++)if(R.listed[k]==i){
                    double* sample=R.samples+((size_t)slot*R.nlisted+k)*7;double r[3],kappa[3];lab(P,t.q,r);lab(P,t.p,kappa);
                    if(lane<3){sample[lane]=r[lane];sample[3+lane]=kappa[lane];}if(lane==0)sample[6]=1.;
                }
            }
            if(held&&i<R.outputs){const long long begun=clock64();__syncwarp();hold_record(P,R,W,h,lane,out);twin_cycles+=clock64()-begun;}
            SPAN_END(SPAN_READOUT,readout);SPAN_BEGIN(windows);
            // The windows ending in the interval: their means while held at their output.
            #pragma unroll
            for(int a=0;a<3;a++)for(int c=0;c<2;c++){
                const int o=next[a];
                if(o>=R.outputs||window_end(R,a,o)>hi)break;
                next[a]++;
                double v=0.;
                if(lane<15&&lane/5==a){
                    const int at=o*R.steps,h=min(R.half[a],at),end=at+h,before=at-h-1;
                    if(t.death<0||t.death>at){
                        const int x=lane-5*a;
                        const double now=end==hi?prefix:ring[ring_at(R,a,end)+x];
                        v=now-(before<0?0.:ring[ring_at(R,a,before)+x]);
                    }
                }
                // v: lane 5a + x holds the window's sum of x; the means divide by its steps (x = 0).
                const double steps=__shfl_sync(FULL,v,5*a);double* close=out+POINT_RECORD+(2*a+c)*CLOSE;
                if(lane<15&&lane/5==a){const int x=lane-5*a;close[x]=x==0?(steps>0.?1.:0.):(steps>0.?v/steps:0.);}
                // The harmonic control of output o along the axis (control.rs `Harmonic::value`): every
                // trajectory's, held or lost, with its κ²'s mean when held.
                if(R.harmonic!=nullptr&&lane==5*a+2){
                    const double* m=R.harmonic+((size_t)o*3+a)*18;double value=0.;
                    for(int p=0;p<3;p++){double s=0.;for(int q=0;q<3;q++)s+=m[3*p+q]*p0[q]+2.*m[9+3*p+q]*q0[q];value+=p0[p]*s;}
                    close[5]=value;close[6]=value*value;close[7]=value*close[2];
                }
            }
            SPAN_END(SPAN_WINDOWS,windows);
        }
    }
    SPAN_BEGIN(summing);
    __syncthreads();
    // The block's sums of the launch's intervals in the trajectories' order: each 128 threads every
    // second interval, 69 of them the point record's entries, 24 the closing windows' means and 18
    // their harmonic controls' sums.
    {
        const int group=threadIdx.x>>7,x=threadIdx.x&127;const size_t stride=(size_t)MOST_INTERVALS*SLOT_RECORD;
        const double* base=R.records+(size_t)first*stride;
        int next[3]={start[0],start[1],start[2]};
        for(int i=R.begin;i<R.end;i++){
            const int hi=i==0?0:min(i*R.steps,last);
            int outputs[3][2],closes[3];
            #pragma unroll
            for(int a=0;a<3;a++){closes[a]=0;while(closes[a]<2&&next[a]<R.outputs&&window_end(R,a,next[a])<=hi)outputs[a][closes[a]++]=next[a]++;}
            if(((i-R.begin)&1)!=group)continue;
            const double* rec=base+(size_t)(i-R.begin)*SLOT_RECORD;
            if(x<POINT_RECORD){
                if(i<R.outputs){double s=0.;for(int w=0;w<count;w++)s+=rec[w*stride+x];R.sums[((size_t)blockIdx.x*R.outputs+i)*RECORD+x]=s;}
            }else if(x<POINT_RECORD+24){
                const int j=x-POINT_RECORD,a=j/8,c=(j/4)%2,m=j%4;
                if(c<closes[a]){
                    double s=0.;for(int w=0;w<count;w++){const double* close=rec+w*stride+POINT_RECORD+(2*a+c)*CLOSE;if(close[0]>0.)s+=close[1+m];}
                    R.sums[((size_t)blockIdx.x*R.outputs+outputs[a][c])*RECORD+MOTION+4*a+m]=s;
                }
            }else if(x<POINT_RECORD+42){
                const int j=x-POINT_RECORD-24,a=j/6,c=(j/3)%2,m=j%3;
                if(c<closes[a]){
                    double s=0.;for(int w=0;w<count;w++)s+=rec[w*stride+POINT_RECORD+(2*a+c)*CLOSE+5+m];
                    R.sums[((size_t)blockIdx.x*R.outputs+outputs[a][c])*RECORD+HARMONIC+3*a+m]=s;
                }
            }
        }
    }
    SPAN_END(SPAN_SUMS,summing);
#ifdef PROFILE
    if(lane==0)W.profile[SPAN_TWIN]+=twin_cycles;
    SPAN_END(SPAN_TOTAL,launch);
    if(active&&lane<SPANS)R.profile[(size_t)slot*SPANS+lane]+=(unsigned long long)W.profile[lane];
#endif
    if(active){
        if(lane==0){
            #pragma unroll
            for(int i=0;i<3;i++){S->q[i]=t.q[i];S->p[i]=t.p[i];}
            S->time=t.time;S->logn=t.logn;S->threshold=t.threshold;S->log_threshold=t.log_threshold;
            S->doppler=t.doppler;S->cluster=t.cluster;S->jumps=t.jumps;S->death=t.death;S->beats=t.beats;
        }
        if(lane<15)S->prefix[lane]=prefix;
        if(lane<MAXM)S->psi[lane]=W.psi[lane];
        if(lane<MAXN)S->twin[lane]=W.twin[lane];
        if(lane==0){
            S->twin_logn=h.logn;S->twin_threshold=h.threshold;S->twin_log_threshold=h.log_threshold;S->twin_cluster=h.cluster;S->twin_jumps=h.jumps;
            R.costs[2*slot]+=(unsigned long long)twin_cycles;R.costs[2*slot+1]+=(unsigned long long)(clock64()-launched);
        }
    }
}

// A wave's task sums merged per run of tasks of one group, in the tasks' order (lib.rs `run`):
// runs[2r] is run r's first task, runs[2r + 1] its tasks.
extern "C" __global__ void merge(const double* sums,const int* runs,int nruns,int outputs,double* out){
    const long long per=(long long)outputs*RECORD,i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nruns*per)return;
    const int r=(int)(i/per);const long long x=i-r*per;double s=0.;
    for(int t=0;t<runs[2*r+1];t++)s+=sums[(runs[2*r]+t)*per+x];
    out[i]=s;
}

// ---------------------------------------------------------------------------------------------
// Checks (tests/gpu.rs): a trajectory's cluster's exact-step components at its point (`exact_components`)
// into X, one warp; a stream's first words and deviates; the structs' sizes.
extern "C" __global__ void exact_probe(const Program P,const State* S,float2* X){
    __shared__ Warp W;const int lane=threadIdx.x&31;if(threadIdx.x>=32)return;
    Traj t;
    #pragma unroll
    for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];}
    t.time=S->time;t.cluster=S->cluster;
    point(P,W,t,lane);exact_components(P,W,P.clusters[t.cluster],lane,X);
}
// A check's exact responses (tests/gpu.rs): a trajectory's cluster's photon tables at its point
// (`exact_responses`) into X, their rows' room after them, one warp; NaN at X[0] where a resolvent is singular.
extern "C" __global__ void response_probe(const Program P,const State* S,float2* X){
    __shared__ Warp W;const int lane=threadIdx.x&31;if(threadIdx.x>=32)return;
    Traj t;
    #pragma unroll
    for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];}
    t.time=S->time;t.cluster=S->cluster;
    point(P,W,t,lane);
    if(!exact_responses(P,W,P.clusters[t.cluster],lane,X,X+P.nnear*MAXN*3*NG)&&lane==0)X[0]=make_float2(__int_as_float(0x7fc00000),0.f);
}
// A check's jump (tests/gpu.rs): a trajectory's state, a jump at its position and time (the first order
// in the Doppler shift, or the exact response where `exact`), the state it leaves.
extern "C" __global__ void jump_probe(const Program P,const Run R,State* S,float2* scratch,int exact){
    __shared__ Warp W;const int lane=threadIdx.x&31;if(threadIdx.x>=32)return;
    Traj t;
    #pragma unroll
    for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];}
    t.time=S->time;t.logn=S->logn;t.threshold=S->threshold;t.log_threshold=S->log_threshold;t.doppler=S->doppler;t.cluster=S->cluster;t.jumps=S->jumps;
    t.death=S->death;t.index=S->index;t.beats=S->beats;
    if(lane<MAXM)W.psi[lane]=S->psi[lane];
    __syncwarp();jump(P,R,W,t,exact!=0,lane,scratch);__syncwarp();
    if(lane<MAXM)S->psi[lane]=W.psi[lane];
    if(lane==0){for(int i=0;i<3;i++)S->p[i]=t.p[i];S->cluster=t.cluster;S->jumps=t.jumps;S->threshold=t.threshold;S->log_threshold=t.log_threshold;S->logn=t.logn;}
}
// Checks (tests/gpu.rs): a stream's first words and deviates; the structs' sizes.
#define OFFSET(T,f) ((u32)(u64)&(((T*)0)->f))
extern "C" __global__ void rng_stream(u64 seed,u32 purpose,u64 trajectory,u32 event,u32* words,double* deviates){
    if(threadIdx.x!=0)return;
    Rng a(seed,purpose,trajectory,event);for(int i=0;i<10;i++)words[i]=a.next_u32();
    Rng b(seed,purpose,trajectory,event);deviates[0]=b.uniform();deviates[1]=b.open_uniform();b.direction(deviates+2);deviates[5]=b.uniform();
}
extern "C" __global__ void layout(u32* out){
    if(threadIdx.x!=0)return;
    out[0]=sizeof(Beam);out[1]=sizeof(Carrier);out[2]=sizeof(Cluster);out[3]=sizeof(Group);out[4]=sizeof(Class);out[5]=sizeof(Photon);
    out[6]=sizeof(Coherent);out[7]=sizeof(Channel);out[8]=sizeof(Program);out[9]=sizeof(State);out[10]=sizeof(Run);
    out[11]=(u32)OFFSET(Program,hm);out[12]=(u32)OFFSET(Program,half_flow);out[13]=(u32)OFFSET(Run,listed);
    out[14]=(u32)OFFSET(State,psi);out[15]=(u32)OFFSET(State,index);
    out[16]=MOST_INTERVALS;out[17]=SLOT_RECORD;out[18]=RECORD;out[19]=WARPS;out[20]=MAXN;out[21]=TWIN;out[22]=MOTION;out[23]=HARMONIC;out[24]=sizeof(Fourth);out[25]=sizeof(NearUnit);
}
