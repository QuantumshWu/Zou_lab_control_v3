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
// One warp per trajectory, each its own block: a warp that has run its trajectory through a launch's intervals leaves,
// and the hardware starts the next slot on its place (an exact response costs as much as hundreds of steps: a
// block of warps would wait for its slowest). A task of the CPU's (lib.rs `TASK`: up to eight consecutive
// trajectories of one group) is eight slots. The warp's lanes share its work: a lane per beam for the fields, a
// lane per matrix entry (e = lane + 32 j, j < SLOTS) for the tables' gathers and the generator, a
// lane per row for the propagator's products and the readout's excited states. The trajectory's
// scalars are uniform over the warp; ψ and the matrices are in its shared memory. A cluster holds
// at most MAXN states: the kernel's bucket (excited states joining clusters would add a larger one,
// with its matrices in shared memory, chosen per solve by the largest cluster).
//
// # Launches
// A launch runs every warp of a wave for its cycle budget (src/gpu/mod.rs keeps it short of the display watchdog), from
// where its trajectory stands, within the MOST_INTERVALS output intervals after the first not yet summed; the trajectories' states, motion windows
// and samples stay in device memory between launches. Each warp runs its trajectory through the
// launch's intervals on its own, its record of each (the output's, the motion windows closing in it)
// into its slot's records; after the launch `task_sums` sums the records per interval over each task in the
// trajectories' order -- the CPU's order within a task -- into the task's sums.

typedef unsigned int u32;
typedef unsigned long long u64;

#define MAXN 8                          // the most states of a cluster of ground states the kernel holds
#define MAXM 16                         // the most states of a cluster its exact step holds (explicit excited states, beats)
#define EXACT_ROUND 1e-7f               // the exact step's exponentials summed to single precision's round-off
#define SLOTS ((MAXN*MAXN+31)/32)       // a lane's entries of an n × n matrix
#define NG 8                            // ground states
#define NE 24                           // excited states
#define MOST_BEAMS 32
#define DBL_EPSILON 2.220446049250313e-16
#define FLT_EPSILON 1.1920929e-7
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
#define PAST_ALL 0xffffffffffffffffull     // a report's interval: the slot has run its task's last (src/gpu/mod.rs `run`)
#define TAU 6.283185307179586
#define PURPOSE_JUMP 3u                 // rng.rs `Purpose::Jump`
// A mirror item's role (`Program::items`), the low bits of an item's fourth word; above them its pair's `Group::first`.
#define ITEM_ADJOINT 1                  // the adjoint of a pair standing for two
#define ITEM_FIRST 2                    // a group between two carriers: its first order's adjoint alone
#define ITEM_KEPT 3                     // the adjoint of what the first order's transfer leaves
#define ITEM_ROLE 3
#define ITEM_PAIR 2
#define MOST_FIRST 25                   // the most field pairs with a first order of a cluster the generator takes in shared memory (`Warp::first`)

// Profiling (examples/profile.rs compiles the kernels with -DPROFILE): the cycles each warp spends in
// the parts of its work, and the advances and jumps it makes, per trajectory into Run::profile.
// Without PROFILE the spans are nothing.
enum Span{SPAN_POINT,SPAN_GENERATOR,SPAN_EXPONENTIAL,SPAN_EVOLVE,SPAN_JUMP,SPAN_EXACT,SPAN_PHOTONS,SPAN_ABSORPTION,SPAN_LOSS,SPAN_STEP,SPAN_READOUT,SPAN_WINDOWS,
    SPAN_TWIN,SPAN_TOTAL,COUNT_ADVANCES,COUNT_JUMPS,COUNT_EXACT,SPAN_JUMPPOINT,SPAN_DRAW,SPAN_EMIT,SPAN_KICK,SPAN_TRANSIENT,SPAN_ITEMS,SPAN_BEAMS,SPAN_MATRICES,SPAN_WEIGHTS,COUNT_NODES,COUNT_PRODUCTS,SPAN_COMPONENTS,SPAN_FORCES,SPAN_CHECKPOINT,COUNT_LAUNCHES,SPANS};  // at most 32: a lane each
#ifdef PROFILE
#define SPAN_BEGIN(name) const long long name=clock64()
#define SPAN_END(span,name) do{if(lane==0)W.profile[span]+=clock64()-name;}while(0)
#define COUNT(span) do{if(lane==0)W.profile[span]++;}while(0)
#define COUNT_BY(span,n) do{if(lane==0)W.profile[span]+=(n);}while(0)
#else
#define SPAN_BEGIN(name) do{}while(0)
#define SPAN_END(span,name) do{}while(0)
#define COUNT(span) do{}while(0)
#define COUNT_BY(span,n) do{}while(0)
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
__device__ __forceinline__ double warp_min(double v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v=fmin(v,__shfl_xor_sync(FULL,v,o));
    return v;
}
__device__ __forceinline__ float warp_max(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(FULL,v,o));
    return v;
}
__device__ __forceinline__ double warp_max(double v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v=fmax(v,__shfl_xor_sync(FULL,v,o));
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
    float half_pole;                    // σ_c/2, the first order's limit on w_l (model.rs `first_shift`)
    int trap;                           // one of the trap's beams (model.rs `trap_beams`): off far from the trap
    int carrier,near;                   // its carrier and its place among the near beams; −1 when not near
    u32 coupled;                        // the ground columns its coupling reaches (bit b)
    float coefficients[MOST_RANK][2];   // c_lk on its carrier's basis
};
struct Carrier{int rank,beams,first;u32 couples;double detuning,wave;};   // its beams: beam_list[first, first + beams); the ground states its folded drives reach (bits); its photons' wave number
// A unit of the light's static shifts (ground.rs `Unit`): its beams unit_beams[first, first + count)
// with their weights γ_l (unit_weights), N, its intensity at the focus s(0) and its weight when taken
// at the focus (model.rs `profile_error`; −1 otherwise). Its intensity s = |Σ_l a_l γ_l|²/N.
struct Unit{int first,count;float norm,focus,profile;int pad;};
// A cluster: its tables' place (node 0 at tables[table], `size` entries per node) and slices, its
// groups, classes and coherent groups, its carriers' photon tables (carrier_tables[carriers + 3c +
// (0 J, 1 J2, 2 T)]), its fourth-order tables (fourths[fourth], −1: folded at second order), its
// states' frame frequencies (frames[frames + x]), where its places' lists of its sparse groups'
// entries start (item_first[positions + 3e + list] for the place e = row·n + col, `Program::items`), its sparse groups' residual
// beats (residuals[beats, beats + nbeats)) and its states' ground positions. For its exact step (`exact`:
// explicit excited states or beats): its ground states n0, its components' parts (1 + its beats, the beats
// omegas[omegas, omegas + parts − 1) and their period after them, 0 without one), its explicit states'
// terms terms[terms, terms + nterms) and dressed photons (the tables at `emission`), its explicit states'
// excited positions (excited_list[excited]), its groups' parts (group_parts[group_parts + g]) and its
// sparse groups' beats' (residual_parts[residual_parts + b]). The near light's shifts ΔE_a^d its exact
// responses' dressing takes (`exact_responses`): bit d·NG + a; the carriers whose static block on it mixes its
// responses (bits; ground.rs `Ground::unmixed`, `carrier_responses`). The field pairs of its groups with a first order:
// first_fields[first_fields, first_fields + nfirst) (`first_table`).
struct Cluster{int n,size,table,h0,groups,ngroups,classes,nclasses,coherent,ncoherent,carriers,fourth,frames,positions,beats,nbeats,
    n0,exact,parts,omegas,terms,nterms,emission,excited,group_parts,residual_parts,first_fields,nfirst;int states[MAXM];u64 shifts;u32 mixed;int pad;};
// A cluster's fourth-order tables (model.rs `FourthLayout`, # Fourth order): their places in its node
// slice per carrier (−1: none), the carriers that drive it carriers[0, on). H^(4)'s quartic part per
// pair (c, d), r_c² r_d² matrices n × n of the monomials E^c*_k1 E^c_k2 E^d*_k3 E^d_k4; per carrier
// ½[R†R, H_g], R†R and YW, r_c² matrices n × n of E^c*_k1 E^c_k2; for the jumps per pair (c, e)
// C_a G_b(ω_c) T^e_k[:, b′] (r_e × 3 axes × ng rows × ng b′ × n), per pair (d, e) P^{de} (r_d r_e × ng × n),
// per carrier R†R on every ground state (r_c² × ng × n) and C_a T^c_k (r_c × 3 × ng × ng).
struct Fourth{int quartic,nquartic,slow,nslow,jumps,njumps,factors,nfactors;u32 bends;int pad[3];};
// One monomial's slab of H^(4) (model.rs `QuarticSlab`): its fields E*_0 E*_1 E_2 E_3 (carrier·MOST_RANK + rank), where it lies,
// where its entries' parts lie (n² of them, −1 for an entry left out) and its lab frequency.
struct Quartic{int fields[4];int at,parts;double nu;};
// One carrier pair's slabs (model.rs `SlowSlab`): R_c1†R_c2 at rr, YW_c1c2 at yw, and their lab frequency.
struct Slow{int c1,c2,rr,yw;double nu;};
// The jumps' W3 at a photon carrier through an inner one (model.rs `JumpSlab`): its table, whether each column has its own source
// alone, and its middle factors (jump_pairs[pairs, pairs + npairs), into `Program::factors`).
struct JumpTable{int photon,inner,at,diagonal,pairs,npairs;};
// A middle factor's table (model.rs `FourthLayout::p`): r_c2 r_c4 × ng × n.
struct Factor{int c2,c4,at,pad;};
// A kept group (model.rs `GroupLayout`): its field pairs pairs[pairs, pairs + npairs), the S and Q
// slices and its entries' count (n² for a group of every entry, row-major; fewer for a sparse one,
// its values per pair in its entries' order); 1 + its first pair's place among the cluster's `first_fields`, its pairs in their order after it, 0 without a first order.
struct Group{int absorbed,emitted,hermitian,pairs,npairs,s,q,count,first,pad;double residual;};
// A field pair of a group with a first order (model.rs `GroupLayout::first`): its group's transfer's place in the cluster's node slice, its kind (bits 0–1 the
// pair's emitted field k1, 2–3 its absorbed k, 4–6 the absorbed carrier, 7–9 the emitted, 10 a Hermitian group's) and its group's beat.
struct FirstField{int transfer,kind;double residual;};
// A jump class (ground.rs `Class`): its photons photons[photons, photons + nphotons) by the row they
// land in and their carrier, the carriers they come from (bits), and where each row's photons start
// (class_rows[rows + r], r up to the target's size).
// Its photons from explicit states directs[direct, direct + ndirect). `beating`: one of its photons has a residual.
struct Class{int target,photons,nphotons,carriers,rows,direct,ndirect,beating;double wave_number;};
// A photon: its carrier, its source column and ground row in the jump tables, its row in the target cluster, ω̃ − ω̃₀.
struct Photon{int carrier,col,to,row;double residual;};
struct Coherent{int channels,count;};
struct Channel{int carrier,col;double residual;};
// A term of a cluster's explicit states (model.rs `exact_terms`): a kept drive (kind 0), a decay cross term
// (1), a decay (2) or a dressing's drive (3); its entry, carrier, part and its mirror's part (the negative
// residual's), its tables' place (a cross term's X_k, and at `at2` its X2_k), a drive's U_k.
struct Term{int kind,row,col,carrier,part,mirror,at,at2;float u[MOST_RANK][2];};
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
    const Fourth* fourths;const Quartic* quartics;const int* quartic_parts;const Slow* slows;const JumpTable* jump_tables;const int* jump_pairs;const Factor* factors;const double* frames;
    // The sparse groups' entries by the place they land at, and what the first order's transfer leaves of any group: per
    // cluster and place three lists in the groups' and pairs' order, items[item_first[positions + 3e + l], item_first[positions + 3e + l + 1]): 0 its pairs' entries;
    // 1 what the transfer leaves there, 2 the mirrors of the other two's entries at (col, row), by their roles (ITEM_*). Each item (its S's and its Q's place in the cluster's
    // node slice -- a kept entry's ΔW^S and ΔW^Q --, its kind, its role and its pair's `Group::first`): the kind's bits 0–1 the pair's emitted field k1, 2–3 its absorbed k, 4–6
    // the absorbed carrier, 7–9 the emitted, 10 a pair standing for two, 12– the beat (`residuals`).
    const int* item_first;const int4* items;const double* residuals;
    const Unit* units;const int* unit_beams;const float2* unit_weights;
    // The exact step's (`Cluster`): the terms, beats, parts, explicit states, direct photons; the excited manifolds
    // levels[level_first[m], level_first[m + 1]) and their widths, their states' mean decay (the ringing's photon, `ringing_photon`).
    const Term* terms;const double* omegas;const int* group_parts;const int* residual_parts;const int* excited_list;const Direct* directs;
    const int* levels;const int* level_first;const double* level_width;
    // The exact response's dressing (`exact_responses`): every beam's kept drives V_l − V_l^f (NE × NG), D (NE × NE),
    // the near units by carrier (near_units[near_first[d], near_first[d + 1])) and their folded fields W^f
    // (unit_couplings, NG columns of NE each), every channel's far channels (d, a) vias[via_first[c·NG + b],
    // via_first[c·NG + b + 1]); H's ground block, its base and per focused beam (NG × NG each), per channel the weight
    // below which a state's term of a mixed block is left out (`carrier_responses`), and per channel (d, a) its units at
    // their focus intensity summed, Σ_u s(0) N W^f[:, a] W^f[:, a]† (lane-major, entry j of row i at [j·NE + i]).
    const float2* kept;const float2* decay;const NearUnit* near_units;const float2* unit_couplings;const int2* vias;const int* via_first;const float2* ground_block;const float* term_floor;const float2* powers;
    const FirstField* first_fields;
    int nbeams,ncarriers,nclusters,focused,nodes,fields,nnear,doppler,loss;
    int weights;                        // where a jump's scratch holds the fourth order's weights
    int nunits;
    int exact_at,transient_at,nlevels,rows_at,between_at;  // the exact step's, the transient's, the exact responses' rows' and the mixed blocks' shifts' room in a trajectory's scratch
    int focused_unit[MOST_FOCUSED],stride[MOST_FOCUSED];   // the focused units (the trap beam's first), the stencil's strides
    int field_carrier[MOST_CARRIERS*MOST_RANK],field_rank[MOST_CARRIERS*MOST_RANK];
    int ground_states[NG],excited_states[NE],ground_cluster[NG],ground_position[NG];
    int near_first[MOST_CARRIERS+1];
    u32 explicit_of[NG],partners[NE];   // per ground state its cluster's explicit states, per excited state its dressing's partners (bits)
    u32 near_focused;                   // the carriers with a focused near unit (bits)
    float hm,light2,allowance,truncation;   // ħ/m (um²/us), twice the light shift's bound, the first order's allowance, the exponentials' truncation
    float ringing;                          // what a coupling within a manifold may move a ringing photon's amplitude by and be left out (model.rs `RINGING_SHARE`)
    float recoil_spread;                    // the least spread of a photon's parts' recoils along a direction their weights move along, of the largest recoil (engine.rs `RECOIL_SPREAD`)
    float axes32[3][3],stiffness32[3];      // the axes and the stiffness below, in single precision
    double hm64,dt;
    double axes[3][3],omega[3],stiffness[3];// the trap's principal axes (rows), ω_i (rad/us), ω_i²/(ħ/m)
    double half_flow[3][3];                 // the flow for dt/2 per axis: cos ωτ, (ħ/m) sin ωτ/ω, (m/ħ) ω sin ωτ
    double region;                          // the trap's region the loss takes (engine.rs `Settings::region`), rad/us
    double lattice_axis[3],lattice_k,lattice_inv_rayleigh,lattice_phase; // the trap's standing wave (model.rs `Lattice`): its axis on the principal axes, k (0: none), 1/z_R, its phase
    double node_phase,grid_phase,beats,horizon;   // the exact step's node phase, a beating grid's (engine.rs `Settings::node_phase`), its beats' share and their horizon
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
    int cluster,jumps,death;            // death: the step it was lost at, −1 while held
    u32 index;int twin_cluster,twin_jumps;float beats;   // beats: the largest bound per unit time of the beats its exact steps left out
    int interval,step;                  // where it stands: the output interval it is in, its last step (−1 before the first)
};
// The held twin (control.rs `Twin`, engine.rs `Engine::held`), none when `u` is null: per cluster k
// its half step's propagator `u` and its H_eff `h` (n_k² entries from blocks[k], row-major) and its
// populations' readout (32 coefficients per density entry from 32·blocks[k], control.rs `readout`);
// per class its jump operators `l` (3 × target × cluster from jumps[class]).
struct Held{const float2* u;const float2* h;const float2* l;const double* readout;const int* blocks;const int* jumps;};
// A launch: a wave's tasks over the output intervals [begin, end).
struct Run{
    State* states;double* rings;double* samples;double* sums;float2* scratch;
    const int* tasks;                   // per task its last output, its trajectories and its first slot
    unsigned long long* profile;        // per slot SPANS counters (with PROFILE)
    double* records;                    // per slot MOST_INTERVALS records of SLOT_RECORD: a ring over the intervals not yet summed
    const double* harmonic;             // per output and lab axis the harmonic control's A and B (control.rs `Harmonic`); none: no control
    unsigned long long* costs;          // in host memory, per slot this launch's cycles of the held twin and in all (lib.rs `Outcome::seconds`), and the interval it stands in
    Held held;
    u64 seed;
    long long budget;                   // the cycles a warp takes in a launch, a step at least (src/gpu/mod.rs)
    int outputs,steps,reach,begin,end;  // the output times, the steps between two, the steps past the last; the first interval not yet summed, the end of the records' room (`task_sums`: of the intervals every slot has passed)
    int half[3],ring[3],ring_at[3];     // per lab axis: the window's half-width, its ring's length (steps) and place (doubles)
    int ring_stride,scratch_stride,nlisted,pad;
    int listed[MOST_LISTED];
};

// ---------------------------------------------------------------------------------------------
// A warp's shared memory.

// A carrier's field at the point: E_k, ∇E_k (per axis), E^D_k and ∇E^D_k (model.rs `CarrierField`).
struct Fields{float2 e[MOST_RANK],grad[MOST_RANK][3],doppler[MOST_RANK],doppler_grad[MOST_RANK][3];};
// A field pair's first order at the point (`first_table`), in Γ: its coefficient on Q at its entries and, at their mirrors, b* of its b on Q†; its gradient's coefficients on Q.
struct FirstPair{float2 on,off,grad[3];};
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
// The exact responses' working space (`exact_responses`): the shifts ΔE_a^d; a mixed block's off-diagonal part and
// eigenvectors, its terms' weights over the columns, its eigenvalues and its states' own energies.
struct BlockSpace{float shift[MOST_CARRIERS*NG];float2 a[MAXN*MAXN],u[MAXN*MAXN],weights[MAXN];double values[MAXN],own[MAXN];};
// A jump's working space.
struct JumpSpace{float2 u[3][MAXN],f[MAXN];float weights[MOST_CLASSES],shares[MOST_BEAMS],projections[MOST_BEAMS],drives[MOST_BEAMS],direct;int beams[MOST_BEAMS],driven[MOST_BEAMS],ndrives;};
struct Warp{
    Point pt;
    float2 psi[MAXM],start[MAXM];       // ψ, ψ at a step's start
    double2 readout[MAXN];              // the readout's Y ψ
    float2 twin[MAXN];                  // the held twin's ψ
    // vec: the exact step's vectors (`half_of`); dressing: the exact responses' shifts ΔE_a^d (`exact_responses`); first: a generator's
    // first orders (`first_table`), clear of m[2].
    union{float2 m[3][MAXN*MAXN];FirstPair first[MOST_FIRST];float2 gradient_a[MOST_BEAMS][3];JumpSpace jump;float4 vec[MAXM/2];BlockSpace block;};
#ifdef PROFILE
    long long profile[SPANS];
#endif
};
static_assert(sizeof(FirstPair)*MOST_FIRST<=2*MAXN*MAXN*sizeof(float2),"a generator's first orders lie clear of W.m[2]");

// The trajectory's scalars (uniform over the warp).
struct Traj{double q[3],p[3],time,logn,threshold,log_threshold;int cluster,jumps,death;u32 index;float beats;};
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

// The C1 cubic Hermite weights with Catmull-Rom's central slopes (one-sided at the grid's ends) on four consecutive
// nodes of a uniform grid of `nodes` points on [0, 1] at s, and their derivatives in s (model.rs `cubic`).
__device__ void cubic(int nodes,float s,int& first,float w[4],float d[4]){
    const float h=(float)(nodes-1);const float x=fminf(fmaxf(s,0.f),1.f)*h;
    const int i=min((int)floorf(x),nodes-2);const float u=x-(float)i;
    const float a=2.f*u*u*u-3.f*u*u+1.f,b=u*u*u-2.f*u*u+u,c=-2.f*u*u*u+3.f*u*u,e=u*u*u-u*u;
    const float da=6.f*u*u-6.f*u,db=3.f*u*u-4.f*u+1.f,dc=-6.f*u*u+6.f*u,de=3.f*u*u-2.f*u;
    if(i==0){first=0;
        w[0]=a-1.5f*b-0.5f*e;w[1]=2.f*b+c;w[2]=-0.5f*b+0.5f*e;w[3]=0.f;
        d[0]=(da-1.5f*db-0.5f*de)*h;d[1]=(2.f*db+dc)*h;d[2]=(-0.5f*db+0.5f*de)*h;d[3]=0.f;}
    else if(i==nodes-2){first=nodes-4;
        w[0]=0.f;w[1]=-0.5f*b+0.5f*e;w[2]=a-2.f*e;w[3]=0.5f*b+c+1.5f*e;
        d[0]=0.f;d[1]=(-0.5f*db+0.5f*de)*h;d[2]=(da-2.f*de)*h;d[3]=(0.5f*db+dc+1.5f*de)*h;}
    else{first=i-1;
        w[0]=-0.5f*b;w[1]=a-0.5f*e;w[2]=c+0.5f*b;w[3]=0.5f*e;
        d[0]=-0.5f*db*h;d[1]=(da-0.5f*de)*h;d[2]=(dc+0.5f*db)*h;d[3]=0.5f*de*h;}
}

// The kinetic shift the first order takes (model.rs `first_shift`): beam l's within half its carrier's nearest pole.
__device__ __forceinline__ float first_shift(const Point& pt,const Beam& b,int l){return fminf(fmaxf(pt.shift[l],-b.half_pole),b.half_pole);}
// The point of the trajectory (model.rs `Model::point`): every beam's amplitude and kinetic shift,
// the focused units' intensities s = |F|²/N (F = Σ_l a_l γ_l), their gradients 2 Re(F* ∇F)/N and
// stencils (and with the trap's light off), the carriers' fields (and with the trap's light off); with FORCE, ∇E^D too, as a generator takes it.
template<bool FORCE>
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
        float2 e=zero2(),o=zero2(),d=zero2(),g[3]={zero2(),zero2(),zero2()},dg[3]={zero2(),zero2(),zero2()};
        for(int j=0;j<C.beams;j++){
            const int l=P.beam_list[C.first+j];const Beam& b=P.beams[l];const float2 x=make_float2(b.coefficients[k][0],b.coefficients[k][1]);const float w=first_shift(pt,b,l);
            const float2 ax=pt.a[l]*x;e+=ax;d+=w*ax;if(!b.trap)o+=ax;
            #pragma unroll
            for(int i=0;i<3;i++){const float2 y=W.gradient_a[l][i]*x;g[i]+=y;if(FORCE)dg[i]+=w*y;}
        }
        Fields& f=pt.fields[c];f.e[k]=e;f.doppler[k]=d;pt.off[c][k]=o;
        #pragma unroll
        for(int i=0;i<3;i++){f.grad[k][i]=g[i];if(FORCE)f.doppler_grad[k][i]=dg[i];}
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

// A place of two tables at the stencil (a photon's J and J2), each place's nodes one batch of loads.
__device__ __forceinline__ void entry_pair(const Program& P,const Point& pt,const float2* x,const float2* y,int size,float2& vx,float2& vy){
    const float* w0=pt.weight[0];vx=vy=zero2();
    for(int o=0;o<(1<<(2*(P.focused-1)));o++){
        const Row r=stencil_row(P,pt,false,o);const size_t at=(size_t)r.index*size;float2 u[4],v[4];
        #pragma unroll
        for(int m=0;m<4;m++){const size_t node=at+(size_t)m*size;u[m]=__ldg(x+node);v[m]=__ldg(y+node);}
        #pragma unroll
        for(int m=0;m<4;m++){const float w=r.w*w0[m];vx+=w*u[m];vy+=w*v[m];}
    }
}
// The same along a polarization, Σ_a e_a x[a·stride], and the first table's derivatives in every s_f (`absorption`).
__device__ __forceinline__ void contracted(const Program& P,const Point& pt,const float2* x,const float2* y,int stride,int size,const float e[3],float2& vx,float2& vy,float2 (&slope)[MOST_FOCUSED]){
    const float* w0=pt.weight[0];const float* s0=pt.slope[0];vx=vy=zero2();
    #pragma unroll
    for(int g=0;g<MOST_FOCUSED;g++)slope[g]=zero2();
    for(int o=0;o<(1<<(2*(P.focused-1)));o++){
        const Row r=stencil_row(P,pt,false,o);const size_t at=(size_t)r.index*size;float2 u[4][3],v[4][3];
        #pragma unroll
        for(int m=0;m<4;m++){
            const size_t node=at+(size_t)m*size;
            #pragma unroll
            for(int a=0;a<3;a++){u[m][a]=__ldg(x+node+a*stride);v[m][a]=__ldg(y+node+a*stride);}
        }
        #pragma unroll
        for(int m=0;m<4;m++){
            const float2 cu=e[0]*u[m][0]+e[1]*u[m][1]+e[2]*u[m][2],cv=e[0]*v[m][0]+e[1]*v[m][1]+e[2]*v[m][2];
            const float w=r.w*w0[m];vx+=w*cu;vy+=w*cv;slope[0]+=(r.w*s0[m])*cu;
            #pragma unroll
            for(int g=1;g<MOST_FOCUSED;g++)if(g<P.focused)slope[g]+=(r.dw[g]*w0[m])*cu;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The generator.

__device__ __forceinline__ bool zero(float2 z){return z.x==0.f&&z.y==0.f;}

// A monomial's E*_0 E*_1 E_2 E_3 at the point (`Quartic::fields`).
__device__ __forceinline__ float2 monomial(const Point& pt,const Quartic& Q){
    auto e=[&](int f){return pt.fields[f/MOST_RANK].e[f%MOST_RANK];};
    return conj(e(Q.fields[0]))*conj(e(Q.fields[1]))*e(Q.fields[2])*e(Q.fields[3]);
}
// The lane's entries of `out` plus a carrier pair's slab at `at` at the point and t (model.rs `fourth_pair`, before the
// cluster's frames): Σ_k1k2 E^c1*_k1 E^c2_k2 M_k1k2 e^{iνt}.
__device__ void fourth_pair(const Program& P,const Warp& W,const Cluster& C,const Entries& E,const Slow& S,int at,double t,int lane,float2 (&out)[SLOTS]){
    const int r1=P.carriers[S.c1].rank,r2=P.carriers[S.c2].rank;const Fields& f1=W.pt.fields[S.c1];const Fields& f2=W.pt.fields[S.c2];
    const float2 beat=residual(S.nu,t);float2 buf[SLOTS],unused[MOST_FOCUSED][SLOTS];
    for(int k1=0;k1<r1;k1++)for(int k2=0;k2<r2;k2++){
        const float2 m=conj(f1.e[k1])*f2.e[k2];if(zero(m))continue;const float2 c=m*beat;
        gather<false>(P,W,false,P.tables+C.table+at+(k1*r2+k2)*E.nn,C.size,E.nn,lane,buf,unused);
        #pragma unroll
        for(int j=0;j<SLOTS;j++)out[j]+=c*buf[j];
    }
}
// The lane's entries of an n × n matrix in the cluster's frames at t: entry (a, b) times e^{i(θ_a − θ_b)t}.
__device__ void in_frames(const Program& P,const Cluster& C,const Entries& E,double t,int lane,float2 (&x)[SLOTS]){
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)&&E.row[j]!=E.col[j])x[j]=x[j]*cis((P.frames[C.frames+E.row[j]]-P.frames[C.frames+E.col[j]])*t);
}
// The metric's coherent part X = Σ R_c1†R_c2 at the lane's entries at the point and t, in the cluster's frames (model.rs `coherent_metric`).
__device__ void coherent_metric(const Program& P,const Warp& W,const Cluster& C,const Entries& E,double t,int lane,float2 (&x)[SLOTS]){
    const Fourth F=P.fourths[C.fourth];
    #pragma unroll
    for(int j=0;j<SLOTS;j++)x[j]=zero2();
    for(int i=0;i<F.nslow;i++){const Slow S=P.slows[F.slow+i];fourth_pair(P,W,C,E,S,S.rr,t,lane,x);}
    in_frames(P,C,E,t,lane,x);
}
// A cluster's fourth-order part of H_eff at the point and t at the lane's entries (model.rs `fourth_hamiltonian`): over its slabs the
// monomial in its beat's phase times the slab, in the cluster's frames; zero for a cluster folded at second order.
__device__ void fourth_hamiltonian(const Program& P,const Warp& W,const Cluster& C,const Entries& E,double t,int lane,float2 (&h)[SLOTS]){
    #pragma unroll
    for(int j=0;j<SLOTS;j++)h[j]=zero2();
    if(C.fourth<0)return;
    const Fourth F=P.fourths[C.fourth];float2 buf[SLOTS],unused[MOST_FOCUSED][SLOTS];double nu=0.;float2 phase=make_float2(1.f,0.f);
    for(int q=0;q<F.nquartic;q++){
        const Quartic Q=P.quartics[F.quartic+q];const float2 m=monomial(W.pt,Q);if(zero(m))continue;
        if(Q.nu!=nu){nu=Q.nu;phase=residual(nu,t);}
        const float2 c=m*phase;
        gather<false>(P,W,false,P.tables+C.table+Q.at,C.size,E.nn,lane,buf,unused);
        #pragma unroll
        for(int j=0;j<SLOTS;j++)h[j]+=c*buf[j];
    }
    in_frames(P,C,E,t,lane,h);
}

// The first order in the kinetic shifts (model.rs, # The photons and the Doppler term): H^(1) = μ Y + ν Y†, Y = Σ ξ_k′k Q_k′k, ξ_k′k = E^d_k′* E^D_k,
// its coefficients joining Φ's on the Q the generator gathers; its gradient λ^S ∇ξ on the same Q. Here all of it in Γ, H ∋ c X being 2i c X in Γ, H ∋ −(i/2) Γ.

// μ and ν of a transfer (λ^S, λ^Q) in Γ (model.rs `first_coefficients`): 2iμ = iλ^S − λ^Q, 2iν = iλ^S* − λ^Q*.
__device__ __forceinline__ void first_coefficients(float2 ls,float2 lq,float2& mu,float2& nu){mu=make_float2(-ls.y-lq.x,ls.x-lq.y);nu=make_float2(ls.y-lq.x,ls.x+lq.y);}
// ∇_a ξ_k′k.
__device__ __forceinline__ float2 xi_slope(const Fields& fe,const Fields& fa,int k1,int k,int a){return conj(fe.grad[k1][a])*fa.doppler[k]+conj(fe.e[k1])*fa.doppler_grad[k][a];}
// A cluster's field pairs' first orders at the point and t -- `phased`, or of their beats' parts without the phase -- into `table` (model.rs `Model::local`, `Model::components`,
// `Model::gradients`), a lane per pair and every pair at once: on Q at the pair's entries, in a Hermitian group μ ξ_k′k + ν ξ_kk′*, between two carriers μ ξ; on Q† at their mirrors,
// in a Hermitian group's pair standing for two μ ξ_kk′ + ν ξ_k′k*, between two carriers ν ξ* at the mirror's beat; none in a pair on the diagonal of k. The gradient's coefficient on Q:
// λ^S ∇_a ξ_k′k, with its adjoint's in a pair standing for two.
__device__ __forceinline__ void first_table(const Program& P,const Warp& W,const Cluster& C,double t,bool phased,int lane,FirstPair* table){
    if(C.nfirst==0)return;
    const float2* tables=P.tables+C.table;const Point& pt=W.pt;
    __syncwarp();
    for(int f=lane;f<C.nfirst;f+=32){
        const FirstField D=P.first_fields[C.first_fields+f];const int k1=D.kind&3,k=(D.kind>>2)&3;const Fields& fa=pt.fields[(D.kind>>4)&7];const Fields& fe=pt.fields[(D.kind>>7)&7];const bool hermitian=(D.kind>>10)&1;
        float2 ls,lq,mu,nu;entry_pair(P,pt,tables+D.transfer,tables+D.transfer+1,C.size,ls,lq);first_coefficients(ls,lq,mu,nu);
        const float2 phase=phased?residual(D.residual,t):make_float2(1.f,0.f),xi=phase*(conj(fe.e[k1])*fa.doppler[k]),lambda=phase*ls,other=hermitian?conj(fe.e[k])*fa.doppler[k1]:zero2();
        FirstPair& F=table[f];F.on=mu*xi+nu*conj(other);F.off=hermitian&&k1==k?zero2():conj(mu*other+nu*conj(xi));
        #pragma unroll
        for(int a=0;a<3;a++){float2 c=lambda*xi_slope(fe,fa,k1,k,a);if(hermitian&&k1<k)c+=conj(lambda*xi_slope(fe,fa,k,k1,a));F.grad[a]=c;}
    }
    __syncwarp();
}

// A cluster's generator at the point and time t: H_eff less its scalar part into W.m[0] (row-major,
// model.rs `Local::hamiltonian`), and in the lane's entries h0 and the gradient of the Hermitian
// part per axis (`Local::gradient`).
struct Local{float2 h0[SLOTS],grad[3][SLOTS];};
__device__ void local(const Program& P,Warp& W,const Cluster& C,const Entries& E,double t,int lane,Local& L){
    // `wide`: a second slot of a lane's entries, beyond five states.
    const int nn=E.nn;const float2* tables=P.tables+C.table;const Point& pt=W.pt;const int D=P.focused;const bool wide=nn>32;
    float2 m[SLOTS],ma[SLOTS],g[SLOTS],ga[SLOTS],buf[SLOTS],slope[MOST_FOCUSED][SLOTS];
    #pragma unroll
    for(int j=0;j<SLOTS;j++){
        m[j]=ma[j]=g[j]=ga[j]=zero2();
        #pragma unroll
        for(int a=0;a<3;a++)L.grad[a][j]=zero2();
    }
    first_table(P,W,C,t,true,lane,W.first);
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
            for(int j=0;j<SLOTS;j++)if(j==0||wide){m[j]+=ee*buf[j];if(twice)ma[j]+=ee*buf[j];}
            #pragma unroll
            for(int a=0;a<3;a++){
                const float2 c=(-both)*(phase*(conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a]));
                #pragma unroll
                for(int j=0;j<SLOTS;j++)if(j==0||wide)L.grad[a][j]+=c*buf[j];
                #pragma unroll
                for(int f=0;f<MOST_FOCUSED;f++)if(f<D){
                    const float2 cf=(-both*pt.grad_s[f][a])*ee;
                    #pragma unroll
                    for(int j=0;j<SLOTS;j++)if(j==0||wide)L.grad[a][j]+=cf*slope[f][j];
                }
            }
            gather<false>(P,W,false,tables+G.q+pi*nn,C.size,nn,lane,buf,slope);
            if(G.first>0){
                const FirstPair& F=W.first[G.first-1+pi];const float2 on=ee+F.on,off=twice?ee+F.off:F.off;
                #pragma unroll
                for(int j=0;j<SLOTS;j++)if(j==0||wide){
                    g[j]+=on*buf[j];ga[j]+=off*buf[j];
                    #pragma unroll
                    for(int a=0;a<3;a++)L.grad[a][j]+=F.grad[a]*buf[j];
                }
            }else{
                #pragma unroll
                for(int j=0;j<SLOTS;j++)if(j==0||wide){g[j]+=ee*buf[j];if(twice)ga[j]+=ee*buf[j];}
            }
        }
    }
    // The sparse groups (fewer entries than places, model.rs `scatter`): each place's own entries, an entry and its mirror's part -- from a pair standing for two its
    // adjoint, from a group between two carriers its first order's -- into the adjoint halves; what the first order's transfer leaves of any group, ½(ξ ΔW^S + h.c.) + (i/2)(ξ ΔW^Q + h.c.),
    // and its gradient; their beats' phases once (in W.m[2], free until the adjoint halves below). A cluster of full groups alone has none.
    if(C.nbeats>0){
        float2* phases=W.m[2];
        __syncwarp();
        for(int b=lane;b<C.nbeats;b+=32)phases[b]=residual(P.residuals[C.beats+b],t);
        __syncwarp();
        #pragma unroll
        for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){
            const int e=lane+32*j,at=C.positions+3*e,kept=P.item_first[at+1],end=P.item_first[at+2];
            #pragma unroll 2
            for(int i=P.item_first[at];i<kept;i++){
                const int4 it=__ldg(P.items+i);
                float2 ds[MOST_FOCUSED];const float2 s=entry_slopes(P,W,tables+it.x,C.size,ds);const float2 q=entry(P,W,tables+it.y,C.size,-1);
                const int k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
                const float2 phase=phases[it.z>>12];const float2 ee=phase*conj(fe.e[k1])*fa.e[k];const bool twice=(it.z>>10)&1;const int first=it.w>>ITEM_PAIR;
                m[j]+=ee*s;g[j]+=ee*q;if(twice){ma[j]+=ee*s;ga[j]+=ee*q;}
                if(first>0){g[j]+=W.first[first-1].on*q;ga[j]+=W.first[first-1].off*q;}
                const float both=twice?2.f:1.f;
                #pragma unroll
                for(int a=0;a<3;a++){
                    L.grad[a][j]+=((-both)*(phase*(conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a])))*s;
                    if(first>0)L.grad[a][j]+=W.first[first-1].grad[a]*q;
                    #pragma unroll
                    for(int f=0;f<MOST_FOCUSED;f++)if(f<D)L.grad[a][j]+=((-both*pt.grad_s[f][a])*ee)*ds[f];
                }
            }
            for(int i=kept;i<end;i++){
                const int4 it=__ldg(P.items+i);const int k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
                const float2 phase=phases[it.z>>12];float2 ws,wq,mu,nu;entry_pair(P,pt,tables+it.x,tables+it.y,C.size,ws,wq);first_coefficients(ws,wq,mu,nu);
                const float2 xi=phase*(conj(fe.e[k1])*fa.doppler[k]);g[j]+=mu*xi;ga[j]+=conj(nu)*xi;ws=phase*ws;
                #pragma unroll
                for(int a=0;a<3;a++)L.grad[a][j]+=ws*xi_slope(fe,fa,k1,k,a);
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
            for(int j=0;j<SLOTS;j++)if(j==0||wide)L.grad[a][j]+=c*slope[f][j];
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

// U = exp(−iHτ) for the generator in W.m[0] (engine.rs `propagate`, on the matrix), ‖H‖ the largest column sum of |re| + |im|:
// the Taylor polynomial P of X = −iHτ/2^s through the term whose bound falls below its share of the truncation, U = P^(2^s) by
// squaring. A square costs a term's product, where a vector's series pays parts × terms; and a part of y = ‖H‖τ/2^s ≤ 2 is
// the largest whose round-off stays the single precision's (its largest term 2, the kernel's series' own choice: at ‖H‖τ = 18,
// 14 products against 43 in sequential parts of ½, U's error 1.6e-6 against 2.8e-6), the first square from ‖H‖τ = 1, where
// it saves two products at no cost to speak of. U into W.m[2].
__device__ void exponential(const Program& P,Warp& W,const Entries& E,double tau,int lane){
    float column=0.f;
    if(lane<E.n)for(int a=0;a<E.n;a++){const float2 h=W.m[0][a*E.n+lane];column+=fabsf(h.x)+fabsf(h.y);}
    const float x=warp_max(column)*fabsf((float)tau);int squares=x>1.f?1:0;float scale=x>1.f?0.5f:1.f;
    while(x*scale>2.f){squares++;scale*=0.5f;}
    const float step=(float)tau*scale,y=x*scale,share=P.truncation*scale;int terms=1;float size=y;while(size>share&&terms<40){terms++;size*=y/(float)terms;}
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
    for(int p=0;p<squares;p++){
        product(W.m[2],W.m[2],E,lane,v);
        __syncwarp();
        #pragma unroll
        for(int j=0;j<SLOTS;j++)if(E.has(j,lane))W.m[2][lane+32*j]=v[j];
        __syncwarp();
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

// The harmonic part's centre (on the principal axes) and weight at a step's start (engine.rs
// `Engine::reference`): one within the trap's region, U_h ≤ D, (2u − 1)/u² beyond it, u = U_h/D, about the
// focus or, in a standing wave, the atom's own site; one without a region.
struct Reference{double c[3],w;};
__device__ Reference reference(const Program& P,const double* q){
    Reference R;R.w=1.;
    #pragma unroll
    for(int i=0;i<3;i++)R.c[i]=0.;
    if(!(P.region>0.))return R;
    double harmonic=0.,share=1.,tangent=0.,along=0.;
    if(P.lattice_k>0.){
        double z=0.;
        #pragma unroll
        for(int i=0;i<3;i++)z+=q[i]*P.lattice_axis[i];
        const double phase=P.lattice_k*z-atan(z*P.lattice_inv_rayleigh)+P.lattice_phase,n=round(phase/(0.5*TAU)),target=n*0.5*TAU-P.lattice_phase;
        // Site n's place (model.rs `Lattice::site`).
        double site=target/P.lattice_k;
        for(;;){const double x=site*P.lattice_inv_rayleigh,dz=(P.lattice_k*site-atan(x)-target)/(P.lattice_k-P.lattice_inv_rayleigh/(1.+x*x));site-=dz;if(fabs(dz)<=8.881784197001252e-16*(fabs(site)+1./P.lattice_k))break;}
        tangent=tan(phase-n*0.5*TAU);share=1./(1.+site*site*P.lattice_inv_rayleigh*P.lattice_inv_rayleigh);along=z-site;
        #pragma unroll
        for(int i=0;i<3;i++)R.c[i]=site*P.lattice_axis[i];
    }
    #pragma unroll
    for(int i=0;i<3;i++){const double x=q[i]-R.c[i]-along*P.lattice_axis[i];harmonic+=P.omega[i]*P.omega[i]*x*x;}
    const double u=0.5*harmonic/P.hm64/(P.region*share)+tangent*tangent;
    R.w=share*(u<=1.?1.:(2.*u-1.)/(u*u));
    return R;
}

// The harmonic flow for τ along each principal axis about the centre c, its part weighted by w (`reference`,
// engine.rs `flow`): (x − c, κ) turned by ω√w τ in phase space; free flight when ω is 0.
__device__ void flow(const Program& P,Traj& t,double tau,const Reference& R){
    const bool half=tau==0.5*P.dt&&R.w==1.;const double scale=R.w==1.?1.:sqrt(R.w);
    #pragma unroll
    for(int i=0;i<3;i++){
        const double w=P.omega[i]*scale;double c,sh,sw;
        if(half){c=P.half_flow[i][0];sh=P.half_flow[i][1];sw=P.half_flow[i][2];}
        else if(w==0.){c=1.;sh=P.hm64*tau;sw=0.;}
        else{double s;sincos(w*tau,&s,&c);sh=P.hm64/w*s;sw=w/P.hm64*s;}
        const double x=t.q[i]-R.c[i],k=t.p[i];t.q[i]=x*c+k*sh+R.c[i];t.p[i]=k*c-x*sw;
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
    const Reference R=reference(P,t.q);
    flow(P,t,0.5*dt,R);
    point<true>(P,W,t,lane);
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
        const double kick=(double)(tau*(P.axes32[i][0]*force[0]+P.axes32[i][1]*force[1]+P.axes32[i][2]*force[2]+(float)R.w*P.stiffness32[i]*(float)(t.q[i]-R.c[i])));
        M.q[i]=t.q[i];M.p[i]=t.p[i]+0.5*kick;t.p[i]+=kick;
    }
    const float n1=apply(W.m[2],W.psi,C.n,lane);t.logn=lh+(double)logf(n1);
    flow(P,t,0.5*dt,R);
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
// s N W^f[:, a] W^f[:, a]†/(z − Ẽ_a − δ_d), Ẽ_a = E_a + Σ_d ΔE_a^d (the shifts in `shift`), an intermediate in the
// cluster k itself at the state's own energy `own`. The units at their focus intensity summed once (`Program::powers`).
__device__ __forceinline__ void self_energy(const Program& P,const Point& pt,const float* shift,int c,int b,double z,double own,int k,u32 p,int lane,float2 (&a)[NE]){
    if(lane>=NE)return;
    for(int v=P.via_first[c*NG+b];v<P.via_first[c*NG+b+1];v++){
        const int2 via=P.vias[v];
        double e=diagonal_at(P,pt,via.y)+P.carriers[via.x].detuning;for(int d=0;d<P.ncarriers;d++)e+=(double)shift[d*NG+via.y];
        const float pole=1.f/(float)((P.ground_cluster[via.y]==k?own:z)-e);
        const float2* m=P.powers+(via.x*NG+via.y)*NE*NE;
        #pragma unroll
        for(int j=0;j<NE;j++){const float2 x=m[j*NE+lane];if(kept_entry(p,lane,j))a[j]+=pole*x;}
        if((P.near_focused>>via.x)&1u)for(int u=P.near_first[via.x];u<P.near_first[via.x+1];u++){
            if(P.near_units[u].focused<0)continue;
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
// at the point into W.block.shift (bit d·NG + a; zero where not taken): ΔE_a^d = −Σ_u s N Re(W^f[:, a]†
// (K − E_a − δ_d)⁻¹ W^f[:, a]) over d's near units, K decoupled from a's cluster's explicit states. Of a state of
// the cluster with another carrier's block mixed, M_d[p, q] = Σ_u s N W^f[:, b_p]† (K − E_q − δ_d)⁻¹ W^f[:, b_q]
// too, the light shift between its places, per carrier in `between` (`carrier_responses`). False when a resolvent
// is singular.
__device__ __noinline__ bool light_shifts(const Program& P,Warp& W,const Cluster& C,int lane,float2* between){
    const Point& pt=W.pt;float* shift=W.block.shift;const int k=P.ground_cluster[C.states[0]];
    if(C.mixed)for(int e=lane;e<P.ncarriers*MAXN*MAXN;e+=32)between[e]=zero2();
    __syncwarp();shift[lane]=0.f;shift[lane+32]=0.f;__syncwarp();
    for(u64 m=C.shifts;m;m&=m-1){
        const int bit=__ffsll((long long)m)-1,d=bit/NG,a=bit-d*NG,q=P.ground_cluster[a]==k?P.ground_position[a]:-1;
        const bool full=q>=0&&(C.mixed&~(1u<<d))!=0u;
        float2 row[NE];resolvent_row(P,pt,P.explicit_of[a],diagonal_at(P,pt,a)+P.carriers[d].detuning,lane,row);
        int at;if(!factor(row,lane,at))return false;
        float2 sum=zero2();
        for(int u=P.near_first[d];u<P.near_first[d+1];u++){
            const float2* w=P.unit_couplings+(u*NG+a)*NE;
            const float2* y=full?(lane<C.n0?P.unit_couplings+(u*NG+C.states[lane])*NE:nullptr):(lane==0?w:nullptr);float2 dot=zero2();
            substitute(row,at,lane<NE?w[lane]:zero2(),lane,[&](int j,float2 x){if(y!=nullptr)dot+=conj(y[j])*x;});
            sum+=unit_weight(pt,P.near_units[u])*dot;
        }
        if(full&&lane<C.n0)between[(d*MAXN+lane)*MAXN+q]=sum;
        if(lane==(full?q:0))shift[bit]=-sum.x;
    }
    __syncwarp();
    return true;
}
// Channel (c, b)'s K + Σ − z at the point (`resolvent_row`, `self_energy`), decoupled from the explicit
// states p, into `rows` lane-major: entry j of row i at rows[32j + i]; `plain`: without Σ.
__device__ __noinline__ void channel_row(const Program& P,const Warp& W,int c,int b,double z,double own,int k,u32 p,bool plain,int lane,float2* rows){
    float2 a[NE];resolvent_row(P,W.pt,p,z,lane,a);if(!plain)self_energy(P,W.pt,W.block.shift,c,b,z,own,k,p,lane,a);
    #pragma unroll
    for(int j=0;j<NE;j++)rows[j*32+lane]=a[j];
}
// The eigenstates of an n × n Hermitian h, its off-diagonal part in `a` (row-major) and its diagonal in `values`, as
// linear.rs `jacobi`: cyclic Jacobi rotations -- each lane k < n its row's entries, single precision off the
// diagonal, which keeps a small mixing's relative accuracy; double precision on it, the hyperfine splitting
// beside its Zeeman gaps -- until the off-diagonal part is its own round-off (`rounding`, or stops falling) or a sweep rotates
// none, a pair left while |h_pq|² is within share² (width² + (h_pp − h_qq)²) (`hermitian_eigen_within`): h = U Λ U†, U into `u`
// (columns), Λ into `values`. In the warp's shared memory (a block's) or the trajectory's scratch (a manifold's).
__device__ __noinline__ void hermitian_eigen(float2* a,float2* u,double* values,int n,float rounding,float share,float width,int lane){
    for(int e=lane;e<n*n;e+=32)u[e]=e%(n+1)==0?make_float2(1.f,0.f):zero2();
    float previous=__int_as_float(0x7f800000);
    for(;;){
        float off=0.f;for(int e=lane;e<n*n;e+=32)off+=norm2(a[e]);
        __syncwarp();off=warp_sum(off);
        if(off<=rounding*rounding||!(off<previous))break;
        previous=off;bool rotated=false;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            // An entry within its share of the round-off stays (the off-diagonal part's sum within `rounding`). Its modulus
            // scaled: a residue's square underflows, and a phase of another modulus would scale U.
            const float2 apq=a[p*n+q];const float big=fmaxf(fabsf(apq.x),fabsf(apq.y));if(big<=rounding/(float)n)continue;
            // A unitary phase on q makes a_pq real and positive; a real rotation in (p, q) then zeroes it.
            const float2 unit=make_float2(apq.x/big,apq.y/big);const float m=big*sqrtf(norm2(unit));
            const float2 ph=make_float2(apq.x/m,apq.y/m);const double dp=values[p],dq=values[q];const float gap=(float)(dq-dp);
            if(m*m<=share*share*(width*width+gap*gap))continue;
            rotated=true;
            const float tau=gap/(2.f*m),t=(tau<0.f?-1.f:1.f)/(fabsf(tau)+sqrtf(1.f+tau*tau)),c=1.f/sqrtf(1.f+t*t),r=t*c;
            if(lane<n){
                const float2 up=u[lane*n+p],uq=u[lane*n+q]*conj(ph);u[lane*n+p]=c*up-r*uq;u[lane*n+q]=r*up+c*uq;
                if(lane!=p&&lane!=q){
                    const float2 x=a[lane*n+p],y=a[lane*n+q]*conj(ph),a1=c*x-r*y,a2=r*x+c*y;
                    a[lane*n+p]=a1;a[lane*n+q]=a2;a[p*n+lane]=conj(a1);a[q*n+lane]=conj(a2);
                }
            }
            if(lane==0){a[p*n+q]=zero2();a[q*n+p]=zero2();values[p]=dp-(double)(t*m);values[q]=dq+(double)(t*m);}
            __syncwarp();
        }
        if(!rotated)break;
    }
}
// Carrier c's responses on the cluster (ground.rs `Ground::dress_light`, `Ground::response`, # The static ground block):
// R_c = Σ_E Σ_b (K + Σ_b(z_E) − z_E + w_l)⁻¹ ŝ_b P_E[b, :], K decoupled from the cluster's explicit states, over the eigen-
// energies z_E = λ_E + δ_c of h_c = H's ground block on the cluster + Σ_{d≠c} d's light shift between its states,
// −½(M_d[p, q] + M_d[q, p]*) (`light_shifts`): the states without far channels through one resolvent per eigen-energy, per
// eigenstate j of Σ_b ŝ_b u_j[b] with u_j[:]*; each with far channels through its own, ŝ_b with its projector's row, its
// intermediates in the cluster at its own energy h_bb + δ_c. Where its block mixes (`Cluster::mixed`) by its eigenstates
// (`hermitian_eigen`), eigenvalues within twice the rotations' round-off one energy, n ε‖h's off-diagonal part and light shifts‖_F in
// single precision: their pair's response differs by that times the resolvent's square, its own round-off. A term whose bound
// moves the generator by less than single precision's round-off is left out (`Program::term_floor`, as ground.rs
// `Mode::skipped` in double). Unmixed, every state its own eigenstate at its own energy (`Ground::unmixed`). Lane i holds
// row i of each system (`factor`) and lane 8a + to sums C_a R_l's entry. False when a resolvent is singular.
__device__ __noinline__ bool carrier_responses(const Program& P,Warp& W,const Cluster& C,int c,int lane,float2* X,float2* rows,const float2* between){
    const Point& pt=W.pt;BlockSpace& B=W.block;const int n=C.n0,k=P.ground_cluster[C.states[0]],nc=P.ncarriers;const Carrier R=P.carriers[c];const u32 p=P.explicit_of[C.states[0]];
    const bool mixed=(C.mixed>>c)&1u;const float2* y=lane<3*NG?P.emission+lane*NE:nullptr;
    float light=0.f;
    if(lane<n){for(int d=0;d<nc;d++)if(d!=c)light+=B.shift[d*NG+C.states[lane]];const double e=diagonal_at(P,pt,C.states[lane])+(double)light;B.values[lane]=e;B.own[lane]=e;}
    float rounding=0.f;
    if(mixed){
        // Of a size above the light shifts': between its states they come by the same solves as on the diagonal, and a
        // pair's by no coupling lies at their round-off.
        float size=light*light;
        for(int e=lane;e<n*n;e+=32){
            const int i=e/n,j=e-i*n;float2 h=zero2();
            if(i!=j){
                const int f=C.states[i]*NG+C.states[j];h=P.ground_block[f];for(int a=0;a<P.focused;a++)h+=pt.s[a]*P.ground_block[(1+a)*NG*NG+f];
                for(int d=0;d<nc;d++)if(d!=c)h-=0.5f*(between[(d*MAXN+i)*MAXN+j]+conj(between[(d*MAXN+j)*MAXN+i]));
                size+=norm2(h);
            }
            B.a[e]=h;
        }
        size=warp_sum(size);rounding=(float)n*(float)FLT_EPSILON*sqrtf(size);
        __syncwarp();hermitian_eigen(B.a,B.u,B.values,n,rounding,0.f,0.f,lane);
    }
    __syncwarp();
    int order[MAXN];
    for(int i=0;i<n;i++){int j=i;if(mixed){const double v=B.values[i];while(j>0&&B.values[order[j-1]]>v){order[j]=order[j-1];j--;}}order[j]=i;}
    for(int b=0;b<R.beams;b++){const int near=P.beams[P.beam_list[R.first+b]].near;if(lane<3*NG)for(int x=0;x<n;x++)X[exact_at(near,x,0,0)+lane]=zero2();}
    for(int first=0,last;first<n;first=last){
        last=first+1;if(mixed)while(last<n&&B.values[order[last]]-B.values[order[first]]<=2.*(double)rounding)last++;
        double z=0.;for(int m=first;m<last;m++)z+=B.values[order[m]];z=z/(double)(last-first)+R.detuning;
        // Its terms: the states of a weight beyond their floor, those without far channels and those with.
        u32 plain,dressed;
        {
            float w2=lane==order[first]?1.f:0.f;
            if(mixed){w2=0.f;if(lane<n)for(int m=first;m<last;m++)w2+=norm2(B.u[lane*n+order[m]]);}
            const int g=lane<n?C.states[lane]:0;const bool term=lane<n&&((R.couples>>g)&1u)&&w2>0.f&&(!mixed||sqrtf(w2)>P.term_floor[c*NG+g]);
            const bool far=term&&P.via_first[c*NG+g+1]>P.via_first[c*NG+g];
            plain=__ballot_sync(FULL,term&&!far);dressed=__ballot_sync(FULL,far);
        }
        // The plain term (bit NG), then each state's with far channels.
        u32 states=0u;for(u32 m=plain;m;m&=m-1)states|=1u<<C.states[__ffs(m)-1];
        for(u32 terms=dressed|(plain?1u<<NG:0u);terms;){
            const bool whole=(terms>>NG)&1u;const int i=whole?-1:__ffs(terms)-1,g=whole?0:C.states[i];
            terms&=whole?~(1u<<NG):terms-1;
            channel_row(P,W,c,g,z,whole?z:B.own[i]+R.detuning,k,p,whole,lane,rows);
            for(int b=0;b<R.beams;b++){
                const int l=P.beam_list[R.first+b];const Beam& beam=P.beams[l];
                if(!(beam.coupled&(whole?states:1u<<g)))continue;
                float2 a[NE];const float w=pt.shift[l];
                #pragma unroll
                for(int j=0;j<NE;j++){a[j]=rows[j*32+lane];if(j==lane)a[j].x+=w;}
                int at;if(!factor(a,lane,at))return false;
                for(int m=whole?first:last-1;m<last;m++){
                    // Its right-hand side and the weights it spreads with over the columns.
                    float2 s=zero2();
                    if(!whole)s=source(P,pt,l,g,p,lane);
                    else for(u32 places=plain;places;places&=places-1){
                        const int q=__ffs(places)-1;
                        if((beam.coupled>>C.states[q])&1u){const float2 x=source(P,pt,l,C.states[q],p,lane);s+=mixed?x*B.u[q*n+order[m]]:x;}
                    }
                    if(mixed&&lane<n){
                        float2 v=zero2();
                        if(whole)v=conj(B.u[lane*n+order[m]]);else for(int e=first;e<last;e++)v+=B.u[i*n+order[e]]*conj(B.u[lane*n+order[e]]);
                        B.weights[lane]=v;
                    }
                    if(mixed)__syncwarp();
                    float2 response=zero2();substitute(a,at,s,lane,[&](int q,float2 x){if(y!=nullptr)response+=y[q]*x;});
                    if(lane<3*NG){
                        if(mixed)for(int x=0;x<n;x++)X[exact_at(beam.near,x,0,0)+lane]+=response*B.weights[x];
                        else X[exact_at(beam.near,whole?order[first]:i,0,0)+lane]=response;
                    }
                }
            }
        }
    }
    __syncwarp();
    return true;
}

// The exact response's photon tables at the point (model.rs `Exact`, ground.rs `Ground::response`):
// X_l[a][to][col] = (C_a R_l(w_l))[to, b] for every near beam l and cluster column (b its ground state)
// at the point's intensities, dressed by the near light (ground.rs, # The near light's dressing): the
// shifts (`light_shifts`), then per carrier (`carrier_responses`). False when a resolvent is singular, as
// `Exact::new` fails (the jump keeps the first order).
__device__ __noinline__ bool exact_responses(const Program& P,Warp& W,const Cluster& C,int lane,float2* X,float2* rows,float2* between){
    if(!light_shifts(P,W,C,lane,between))return false;
    for(int c=0;c<P.ncarriers;c++)if(!carrier_responses(P,W,C,c,lane,X,rows,between))return false;
    return true;
}

// The fourth order's weights of a jump's photons (model.rs `fourth_photons`): per jump table Σ of its middle factors' forms
// Σ E^c2*_k3 E^c4_k4 P_k3k4 on every ground state b′ × the cluster (ng × n) at weights[table·NG·n], the forms after them.
__device__ void fourth_weights(const Program& P,const Warp& W,const Cluster& C,int lane,float2* weights){
    const Fourth F=P.fourths[C.fourth];const float2* tables=P.tables+C.table;const int size=NG*C.n;const Point& pt=W.pt;float2* forms=weights+F.njumps*size;
    for(int i=lane;i<F.nfactors*size;i+=32){
        const int which=i/size,at=i-which*size;const Factor M=P.factors[F.factors+which];const int r2=P.carriers[M.c2].rank,r4=P.carriers[M.c4].rank;float2 w=zero2();
        for(int k3=0;k3<r2;k3++)for(int k4=0;k4<r4;k4++){
            const float2 c=conj(pt.fields[M.c2].e[k3])*pt.fields[M.c4].e[k4];if(zero(c))continue;
            w+=c*entry(P,W,tables+M.at+(k3*r4+k4)*size+at,C.size,-1);
        }
        forms[i]=w;
    }
    __syncwarp();
    for(int i=lane;i<F.njumps*size;i+=32){
        const int table=i/size,at=i-table*size;const JumpTable J=P.jump_tables[F.jumps+table];float2 w=zero2();
        for(int m=0;m<J.npairs;m++)w+=forms[P.jump_pairs[J.pairs+m]*size+at];
        weights[i]=w;
    }
    __syncwarp();
}
// A photon's W3 part along an axis (model.rs `fourth_photons`): −C W3^c[to, y], over the tables at its carrier's frequency Σ_k E_k Σ_b′
// the table from each of the column's source places b through its weight (b′, b) -- a diagonal table's the column's own.
__device__ float2 fourth_photon(const Program& P,const Warp& W,const Cluster& C,const Photon& ph,int a,const float2* weights){
    const Fourth F=P.fourths[C.fourth];const float2* tables=P.tables+C.table;const int n=C.n,size=NG*n;float2 z=zero2();
    for(int i=0;i<F.njumps;i++){
        const JumpTable J=P.jump_tables[F.jumps+i];if(J.photon!=ph.carrier)continue;
        const int s=J.diagonal?1:n,block=3*NG*NG*n*s,rank=P.carriers[J.inner].rank;const float2* weight=weights+i*size;const Fields& fi=W.pt.fields[J.inner];
        for(int k=0;k<rank;k++){
            const float2 e=fi.e[k];if(zero(e))continue;float2 v=zero2();
            for(int bp=0;bp<NG;bp++){
                const float2* x=tables+J.at+k*block+(((a*NG+ph.to)*NG+bp)*n+ph.col)*s;
                if(J.diagonal)v+=entry(P,W,x,C.size,-1)*weight[bp*n+ph.col];
                else for(int b=0;b<n;b++)v+=entry(P,W,x+b,C.size,-1)*weight[bp*n+b];
            }
            z-=e*v;
        }
    }
    return z;
}
// The state the fourth order's carriers' photons act on (model.rs `photons`): ψ − ½Xψ into W.start, X the metric's coherent part
// (`coherent_metric`), through W.m[0] -- before the jump's space takes it.
__device__ void bent_state(const Program& P,Warp& W,const Cluster& C,double t,int lane){
    const Entries E(C.n,lane);float2 x[SLOTS];
    coherent_metric(P,W,C,E,t,lane,x);
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane))W.m[0][lane+32*j]=x[j];
    __syncwarp();
    if(lane<C.n){float2 s=zero2();for(int b=0;b<C.n;b++)s+=W.m[0][lane*C.n+b]*W.psi[b];W.start[lane]=W.psi[lane]-0.5f*s;}
    __syncwarp();
}
__device__ void photon_matrices(const Program& P,const Warp& W,const Cluster& C,int lane,const float2* X,float2* out,float2* weights){
    const float2* tables=P.tables+C.table;const int first=P.classes[C.classes].photons;
    const Class last=P.classes[C.classes+C.nclasses-1];const int count=3*(last.photons+last.nphotons-first);
    if(C.fourth>=0)fourth_weights(P,W,C,lane,weights);
    for(int j=lane;j<count;j+=32){
        const int a=j%3;const Photon ph=P.photons[first+j/3];const Carrier R=P.carriers[ph.carrier];float2 m=zero2();
        if(X==nullptr){
            const Fields& f=W.pt.fields[ph.carrier];const int j1=P.carrier_tables[C.carriers+3*ph.carrier],j2=P.carrier_tables[C.carriers+3*ph.carrier+1];
            #pragma unroll
            for(int k=0;k<MOST_RANK;k++)if(k<R.rank){const int at=((k*3+a)*NG+ph.to)*C.n0+ph.col;float2 v1,v2;entry_pair(P,W.pt,tables+j1+at,tables+j2+at,C.size,v1,v2);m+=f.e[k]*v1-f.doppler[k]*v2;}
        }else for(int b=0;b<R.beams;b++){const int l=P.beam_list[R.first+b];m+=X[exact_at(P.beams[l].near,ph.col,a,ph.to)]*W.pt.a[l];}
        // On ψ, the fourth order's carriers' on the bent state (`bent_state`), and its W3 part on ψ.
        float2 z=m*(C.fourth>=0&&((P.fourths[C.fourth].bends>>ph.carrier)&1u)?W.start[ph.col]:W.psi[ph.col]);
        if(C.fourth>=0)z+=fourth_photon(P,W,C,ph,a,weights)*W.psi[ph.col];
        out[j]=z;
    }
    __syncwarp();
}
// The amplitude u_a of a class at a row of its target cluster (model.rs `Model::photons`): over its
// photons landing in that row, e^{iω̃t} M, M the photon's matrix on its state (`photon_matrices`); less over its
// explicit states' photons there e^{iω̃t} ψ_e ĉ_a[to] (their dressed photons at the stencil).
__device__ float2 amplitude(const Program& P,const Warp& W,const Cluster& C,const Class& K,int a,int row,double t,const float2* M){
    const int first=P.classes[C.classes].photons,end=P.class_rows[K.rows+row+1];float2 u=zero2();
    for(int i=P.class_rows[K.rows+row];i<end;i++){const float2 m=M[3*(i-first)+a];u+=K.beating?residual(P.photons[i].residual,t)*m:m;}
    for(int i=0;i<K.ndirect;i++){
        const Direct d=P.directs[K.direct+i];if(d.row!=row)continue;
        u-=residual(d.residual,t)*W.psi[d.state]*entry(P,W,P.tables+C.table+C.emission+((d.state-C.n0)*3+a)*NG+d.to,C.size,-1);
    }
    return u;
}

// The picked class's absorbed light along the photon's polarization e (model.rs `Photons::gradient`
// and `Photons::shares`, contracted with e), from one pass over its photons' jump tables. In passes of the class's carriers
// (ascending, as many as the warp's lanes hold, one in nearly every scene) a lane per (carrier, target row, k), a row's k adjacent; of a lane, over its
// photons, A_k = Σ e^{iω̃t} ψ[from] Σ_a e_a J_k[a][to, from], B_k the same of J2_k, and the gradient's part
// through the intensities E_k Σ_f ∂_f A_k ∇s_f, whose Im⟨f|·⟩ is the exact conditional mean's correction
// (`mean`, still to be divided by ‖f‖²). Each beam l of its carriers in their order, on a row's first lane z_l = Σ_k c_lk (A_k − w_l B_k) --
// beyond the first order the exact responses' Σ e^{iω̃t} ψ[from] Σ_a e_a (C_a R_l)[to, from] --: the gradient's part
// through the field ∇a_l z_l, its share Σ_rows ‖a_l z_l‖² into J.shares, its projection on the photon J.f, Re⟨f|a_l z_l⟩ (still
// to be divided by ‖f‖²), into J.projections, and J.beams -- of a plane wave ∇a_l = i k_l a_l, its part of `mean` k_l times its projection --; returns how many; and the
// explicit states' part into J.direct, J.drives and J.driven.
__device__ int absorption(const Program& P,Warp& W,const Cluster& C,const Class& K,int nt,const double e[3],double t,const float2* X,int lane,float mean[3]){
    const float2* tables=P.tables+C.table;const Point& pt=W.pt;JumpSpace& J=W.jump;const int D=P.focused;
    const float ea[3]={(float)e[0],(float)e[1],(float)e[2]};
    #pragma unroll
    for(int x=0;x<3;x++)mean[x]=0.f;
    int count=0;
    for(int from=0,to;from<P.ncarriers;from=to){
        SPAN_BEGIN(items);
        // The pass's carriers and the lane's place among them.
        int c=-1,row=0,k=0,lanes=0;
        for(to=from;to<P.ncarriers;to++){
            if(!((K.carriers>>to)&1))continue;
            const int rank=P.carriers[to].rank,size=nt*rank;if(lanes+size>32)break;
            if(c<0&&lane<lanes+size){c=to;row=(lane-lanes)/rank;k=lane-lanes-row*rank;}
            lanes+=size;
        }
        // The lane's photons: the carrier's run within the row's.
        float2 A=zero2(),B=zero2(),d[MOST_FOCUSED];
        #pragma unroll
        for(int f=0;f<MOST_FOCUSED;f++)d[f]=zero2();
        int lo=0,hi=0;
        if(c>=0){
            lo=P.class_rows[K.rows+row];const int end=P.class_rows[K.rows+row+1];
            while(lo<end&&P.photons[lo].carrier<c)lo++;
            hi=lo;while(hi<end&&P.photons[hi].carrier==c)hi++;
            const int j1=P.carrier_tables[C.carriers+3*c],j2=P.carrier_tables[C.carriers+3*c+1];
            for(int i=lo;i<hi;i++){
                const Photon ph=P.photons[i];
                const float2 phase=residual(ph.residual,t)*W.psi[ph.col];
                const int at=(k*3*NG+ph.to)*C.n0+ph.col;float2 v1,v2,slope[MOST_FOCUSED];
                contracted(P,pt,tables+j1+at,tables+j2+at,NG*C.n0,C.size,ea,v1,v2,slope);
                A+=phase*v1;B+=phase*v2;
                #pragma unroll
                for(int f=0;f<MOST_FOCUSED;f++)if(f<D)d[f]+=phase*slope[f];
            }
            const float2 ek=pt.fields[c].e[k],fr=J.f[row];
            #pragma unroll
            for(int x=0;x<3;x++){
                float2 g=zero2();
                #pragma unroll
                for(int f=0;f<MOST_FOCUSED;f++)if(f<D)g+=pt.grad_s[f][x]*(ek*d[f]);
                mean[x]+=fr.x*g.y-fr.y*g.x;
            }
        }
        SPAN_END(SPAN_ITEMS,items);SPAN_BEGIN(beamsp);
        for(int q=from;q<to;q++){
            if(!((K.carriers>>q)&1))continue;const Carrier R=P.carriers[q];const bool mine=c==q,first=mine&&k==0;
            for(int j=0;j<R.beams;j++){
                const int l=P.beam_list[R.first+j];const Beam& b=P.beams[l];const bool focused=b.inv_w2!=0.f;float2 al=pt.a[l],slope[3];
                if(focused)field(b,pt.r,al,slope);
                // z_l on a row's first lane.
                float2 g=zero2();
                if(X==nullptr){
                    const float wl=first_shift(pt,b,l);
                    if(mine)g=make_float2(b.coefficients[k][0],b.coefficients[k][1])*(A-wl*B);
                    const float2 y=g;
                    for(int r=1;r<R.rank;r++)g+=make_float2(__shfl_down_sync(FULL,y.x,r),__shfl_down_sync(FULL,y.y,r));
                }else if(first)for(int i=lo;i<hi;i++){
                    const Photon ph=P.photons[i];float2 z=zero2();
                    #pragma unroll
                    for(int a=0;a<3;a++)z+=ea[a]*X[exact_at(b.near,ph.col,a,ph.to)];
                    g+=residual(ph.residual,t)*W.psi[ph.col]*z;
                }
                float v=0.f,on=0.f;
                if(first){
                    const float2 z=al*g,fr=J.f[row];v=norm2(z);on=fr.x*z.x+fr.y*z.y;
                    if(focused){
                        #pragma unroll
                        for(int x=0;x<3;x++){const float2 dz=slope[x]*g;mean[x]+=fr.x*dz.y-fr.y*dz.x;}
                    }
                }
                if(!focused){
                    #pragma unroll
                    for(int x=0;x<3;x++)mean[x]+=b.wave[x]*on;
                }
                const float w=warp_sum(v),w1=warp_sum(on);if(lane==0){J.shares[count]=w;J.projections[count]=w1;J.beams[count]=l;}count++;
            }
        }
        SPAN_END(SPAN_BEAMS,beamsp);
    }
    #pragma unroll
    for(int x=0;x<3;x++)mean[x]=warp_sum(mean[x]);
    __syncwarp();
    // The explicit states' part of the photon (`Photons::absorption`): its weight along e, Σ_rows |Σ e^{iω̃t}
    // ψ_e Σ_a e_a ĉ_a[to]|², and the beams' weights: each explicit state's weight in the photon shared by the drives into
    // it, Σ |a_l Σ_k c_lk U_k ψ_b|², in the drives' and their carriers' beams' order.
    float total=0.f;int beams[MOST_BEAMS];float drive[MOST_BEAMS];int nb=0;
    if(K.ndirect>0){
        float2 v=zero2();float2 part[MAXM];float weight[MAXM],into[MAXM];const int ne=C.n-C.n0;
        for(int s=0;s<ne;s++){part[s]=zero2();into[s]=0.f;}
        if(lane<nt)for(int i=0;i<K.ndirect;i++){
            const Direct d=P.directs[K.direct+i];if(d.row!=lane)continue;float2 z=zero2();
            for(int a=0;a<3;a++)z+=ea[a]*entry(P,W,tables+C.emission+((d.state-C.n0)*3+a)*NG+d.to,C.size,-1);
            const float2 c=residual(d.residual,t)*W.psi[d.state]*z;v+=c;part[d.state-C.n0]+=c;
        }
        total=warp_sum(lane<nt?norm2(v):0.f);
        for(int s=0;s<ne;s++)weight[s]=warp_sum(lane<nt?norm2(part[s]):0.f);
        for(int pass=0;pass<2;pass++)for(int i=0;i<C.nterms;i++){
            const Term T=P.terms[C.terms+i];if(T.kind!=0)continue;const Carrier Rc=P.carriers[T.carrier];const int s=T.row-C.n0;
            for(int j=0;j<Rc.beams;j++){
                const int l=P.beam_list[Rc.first+j];const Beam& b=P.beams[l];float2 z=zero2();
                for(int k=0;k<Rc.rank;k++)z+=make_float2(b.coefficients[k][0],b.coefficients[k][1])*make_float2(T.u[k][0],T.u[k][1]);
                const float d=norm2(z*pt.a[l]*W.psi[T.col]);
                if(pass==0){into[s]+=d;continue;}
                int at=0;while(at<nb&&beams[at]!=l)at++;
                if(at==nb){beams[nb]=l;drive[nb]=0.f;nb++;}
                if(into[s]>0.f)drive[at]+=weight[s]*d/into[s];
            }
        }
    }
    __syncwarp();
    if(lane==0){J.direct=total;J.ndrives=nb;for(int i=0;i<nb;i++){J.drives[i]=drive[i];J.driven[i]=beams[i];}}
    __syncwarp();
    return count;
}

// The bare state's slow part Π = 1 + ½X + YW at the lane's entries at the point and t in the cluster's frames, with the metric's
// coherent part X it holds (model.rs `slow_and_metric`).
__device__ void slow_and_metric(const Program& P,const Warp& W,const Cluster& C,const Entries& E,double t,int lane,float2 (&x)[SLOTS],float2 (&pi)[SLOTS]){
    const Fourth F=P.fourths[C.fourth];
    coherent_metric(P,W,C,E,t,lane,x);
    #pragma unroll
    for(int j=0;j<SLOTS;j++)pi[j]=zero2();
    for(int i=0;i<F.nslow;i++){const Slow S=P.slows[F.slow+i];fourth_pair(P,W,C,E,S,S.yw,t,lane,pi);}
    in_frames(P,C,E,t,lane,pi);
    #pragma unroll
    for(int j=0;j<SLOTS;j++)pi[j]=pi[j]+0.5f*x[j]+make_float2(E.has(j,lane)&&E.row[j]==E.col[j]?1.f:0.f,0.f);
}
// ψ ← Π ψ/√A (W.psi, normalized before): A = ‖Π ψ‖².
__device__ float slow_part(Warp& W,const Entries& E,const float2 (&pi)[SLOTS],int lane){
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane))W.m[0][lane+32*j]=pi[j];
    __syncwarp();
    return apply(W.m[0],W.psi,E.n,lane);
}
// The fourth order's start from a ringing photon's landing (engine.rs `slow_start`; the initial states have theirs from the
// CPU): ψ ← Π f/√A, the threshold θ ← θ/A, A = ‖Π f‖². No draws.
__device__ void slow_start(const Program& P,Warp& W,Traj& t,int lane){
    const Cluster C=P.clusters[t.cluster];if(C.fourth<0)return;
    const Entries E(C.n,lane);float2 x[SLOTS],pi[SLOTS];
    slow_and_metric(P,W,C,E,t.time,lane,x,pi);
    t.threshold/=(double)slow_part(W,E,pi,lane);t.log_threshold=log(t.threshold);
}

// A real orthonormal pair across the unit vector n (engine.rs `across`).
__device__ __forceinline__ void across(const double n[3],double pair[2][3]){
    const bool x=fabs(n[0])<=fabs(n[1])&&fabs(n[0])<=fabs(n[2]),y=!x&&fabs(n[1])<=fabs(n[2]);
    const double e0=x?0.:y?-n[2]:n[1],e1=x?n[2]:y?0.:-n[0],e2=x?-n[1]:y?n[0]:0.,l=1./sqrt(e0*e0+e1*e1+e2*e2);
    pair[0][0]=e0*l;pair[0][1]=e1*l;pair[0][2]=e2*l;
    pair[1][0]=n[1]*pair[0][2]-n[2]*pair[0][1];pair[1][1]=n[2]*pair[0][0]-n[0]*pair[0][2];pair[1][2]=n[0]*pair[0][1]-n[1]*pair[0][0];
}

// A quantum jump at the trajectory's position and time (engine.rs `Engine::jump`), with the exact
// response when `exact`: the class and the cluster it lands in, the photon's direction from the
// dipole pattern, its polarization, the recoils, and the state the photon leaves.
// The place x falls in among the weights J.weights[0, n): the last where it is past them by round-off.
__device__ __forceinline__ int picked(const JumpSpace& J,int n,double x){
    for(int c=0;c<n;c++){const double w=(double)J.weights[c];if(x<w)return c;x-=w;}
    return n-1;
}
// A jump's first draws (engine.rs `jump`), each class's weight in J.weights and their total positive:
// the class, the next threshold, the polarization's and the absorbed beam's uniforms -- before the
// direction's rejection loop, so that a twin on the same stream keeps its thresholds and choices.
struct Draws{int pick;double threshold,polarization,absorbed;};
__device__ Draws draw(Rng& rng,const JumpSpace& J,int nclasses,double total){
    Draws d;const double x=rng.uniform()*total;d.threshold=rng.open_uniform();d.polarization=rng.uniform();d.absorbed=rng.uniform();
    d.pick=picked(J,nclasses,x);
    return d;
}
// The photon of the picked class's amplitudes J.u (nt rows): its direction n from the dipole pattern
// tr G − n·Re(G)·n of G_ab = ⟨u_a|u_b⟩, its polarization e across it from ‖Σ_a e_a u_a‖² on a real
// orthonormal pair, f = Σ_a e_a u_a into J.f; returns ‖f‖².
__device__ float emit(Rng& rng,JumpSpace& J,int nt,double polarization,int lane,double n[3],double e[3]){
    const bool row=lane<nt;float2 u[3]={zero2(),zero2(),zero2()};
    if(row){
        #pragma unroll
        for(int a=0;a<3;a++)u[a]=J.u[a][lane];
    }
    const double g00=(double)warp_sum(norm2(u[0])),g11=(double)warp_sum(norm2(u[1])),g22=(double)warp_sum(norm2(u[2])),trace=g00+g11+g22;
    const double g01=(double)warp_sum(u[0].x*u[1].x+u[0].y*u[1].y),g02=(double)warp_sum(u[0].x*u[2].x+u[0].y*u[2].y),g12=(double)warp_sum(u[1].x*u[2].x+u[1].y*u[2].y);
    for(;;){
        rng.direction(n);
        const double q=n[0]*n[0]*g00+n[1]*n[1]*g11+n[2]*n[2]*g22+2.*(n[0]*n[1]*g01+n[0]*n[2]*g02+n[1]*n[2]*g12);
        if(rng.uniform()*trace<=trace-q)break;
    }
    double pair[2][3];across(n,pair);float weights[2];float2 f[2];
    #pragma unroll
    for(int j=0;j<2;j++){
        f[j]=zero2();
        #pragma unroll
        for(int a=0;a<3;a++)f[j]+=(float)pair[j][a]*u[a];
        weights[j]=warp_sum(norm2(f[j]));
    }
    const bool first=polarization*((double)weights[0]+(double)weights[1])<(double)weights[0];
    #pragma unroll
    for(int a=0;a<3;a++)e[a]=first?pair[0][a]:pair[1][a];
    if(row)J.f[lane]=first?f[0]:f[1];
    __syncwarp();
    return first?weights[0]:weights[1];
}

// One of Jacobi's rotations at the place (p, q) of a symmetric matrix, r the third (`on_range`), and of the vectors v (columns).
__device__ __forceinline__ void rotate(double& app,double& aqq,double& apq,double& arp,double& arq,double (&v)[3][3],int p,int q,double round){
    if(!(fabs(apq)>round))return;
    const double theta=0.5*(aqq-app)/apq,t=(theta<0.?-1.:1.)/(fabs(theta)+sqrt(theta*theta+1.)),c=1./sqrt(t*t+1.),s=t*c;
    app-=t*apq;aqq+=t*apq;apq=0.;
    const double x=arp,y=arq;arp=c*x-s*y;arq=s*x+c*y;
    #pragma unroll
    for(int k=0;k<3;k++){const double a=v[k][p],b=v[k][q];v[k][p]=c*a-s*b;v[k][q]=s*a+c*b;}
}
// β = C⁺ g on the range of a positive semidefinite C (engine.rs `symmetric`, `part_weights`: the directions above its round-off along which the parts on spread by
// more than `spread`: a lane's `on` and recoil less their mean `off`, the explicit part's `onx`, its recoil's mean zero, so less `centre`; `sum` their weight).
// Where the adjugate's determinant holds single precision -- its round-off is at most 3 ε tr(C)³, and with it the solution's
// relative error, which the parts' single-precision shares cannot tell from theirs -- and every direction spreads that far (each direction's spread is at least the root of its eigenvalue over
// their weight, and the smallest eigenvalue at least 4 det/tr²), the adjugate's; else Jacobi's rotations
// to double precision, in the cycle of the places (β does not depend on the order; the CPU's rotates the largest first).
__device__ __forceinline__ void on_range(double c00,double c01,double c02,double c11,double c12,double c22,const double g[3],double on,const double off[3],double onx,const double centre[3],double sum,double spread,double beta[3]){
    const double a00=c11*c22-c12*c12,a01=c02*c12-c01*c22,a02=c01*c12-c02*c11,a11=c00*c22-c02*c02,a12=c01*c02-c00*c12,a22=c00*c11-c01*c01;
    const double det=c00*a00+c01*a01+c02*a02,trace=c00+c11+c22;
    if(det*FLT_EPSILON>3.*DBL_EPSILON*trace*trace*trace&&4.*det>trace*trace*sum*spread*spread){
        const double inverse=1./det;
        beta[0]=inverse*(a00*g[0]+a01*g[1]+a02*g[2]);beta[1]=inverse*(a01*g[0]+a11*g[1]+a12*g[2]);beta[2]=inverse*(a02*g[0]+a12*g[1]+a22*g[2]);
        return;
    }
    double v[3][3]={{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}};
    for(int rotations=0;rotations<64;rotations+=3){
        const double round=DBL_EPSILON*(fabs(c00)+fabs(c11)+fabs(c22));
        if(!(fmax(fmax(fabs(c01),fabs(c02)),fabs(c12))>round))break;
        rotate(c00,c11,c01,c02,c12,v,0,1,round);rotate(c00,c22,c02,c01,c12,v,0,2,round);rotate(c11,c22,c12,c01,c02,v,1,2,round);
    }
    const double floor_=64.*DBL_EPSILON*(fabs(c00)+fabs(c11)+fabs(c22)),values[3]={c00,c11,c22};  // engine.rs `SHARE_ROUNDING`
    #pragma unroll
    for(int a=0;a<3;a++)beta[a]=0.;
    #pragma unroll
    for(int i=0;i<3;i++){
        const double along=v[0][i]*off[0]+v[1][i]*off[1]+v[2][i]*off[2],across=v[0][i]*centre[0]+v[1][i]*centre[1]+v[2][i]*centre[2];
        if(!(values[i]>floor_&&fmax(warp_max(on>0.?fabs(along):0.),onx>0.?fabs(across):0.)>spread))continue;
        const double p=(v[0][i]*g[0]+v[1][i]*g[1]+v[2][i]*g[2])/values[i];
        #pragma unroll
        for(int a=0;a<3;a++)beta[a]+=p*v[a][i];
    }
}
// The weights of a photon's parts (engine.rs `part_weights`), a lane per beam: its share s and its recoil's mean mu (zeros on a lane
// without a beam) and the explicit states' share sx (its recoil's mean zero), n parts in all, moved to the mean `target`, their directions those they spread along by more than `least`
// of the largest recoil: the lane's q and the explicit part's qx; false where no part has a share.
__device__ __forceinline__ bool part_weights(double s,const double mu[3],double sx,const double target[3],int n,double least,double& q,double& qx){
    q=qx=0.;const double total=warp_sum(s)+sx;if(!(total>0.))return false;
    const double scale=1./total;q=s*scale;qx=sx*scale;double on=q,onx=qx;
    const double spread=least*sqrt(warp_max(mu[0]*mu[0]+mu[1]*mu[1]+mu[2]*mu[2]));
    for(int it=0;it<n;it++){
        const double sum=warp_sum(on)+onx;if(!(sum>0.))break;
        const double inverse=1./sum;double centre[3],g[3],off[3];
        #pragma unroll
        for(int a=0;a<3;a++){centre[a]=inverse*warp_sum(on*mu[a]);g[a]=target[a]-warp_sum(q*mu[a]);off[a]=mu[a]-centre[a];}
        double beta[3];
        on_range(warp_sum(on*off[0]*off[0])+onx*centre[0]*centre[0],warp_sum(on*off[0]*off[1])+onx*centre[0]*centre[1],warp_sum(on*off[0]*off[2])+onx*centre[0]*centre[2],
            warp_sum(on*off[1]*off[1])+onx*centre[1]*centre[1],warp_sum(on*off[1]*off[2])+onx*centre[1]*centre[2],warp_sum(on*off[2]*off[2])+onx*centre[2]*centre[2],g,on,off,onx,centre,sum,spread,beta);
        const double d=on*(beta[0]*off[0]+beta[1]*off[1]+beta[2]*off[2]),dx=-onx*(beta[0]*centre[0]+beta[1]*centre[1]+beta[2]*centre[2]);
        const bool leaving=on>0.&&d<0.,leaving_x=onx>0.&&dx<0.;
        const double alpha=fmin(warp_min(leaving?-q/d:1.),leaving_x?-qx/dx:1.);
        q+=alpha*d;qx+=alpha*dx;
        if(leaving&&q<=-4.*DBL_EPSILON*alpha*d){q=0.;on=0.;}
        if(leaving_x&&qx<=-4.*DBL_EPSILON*alpha*dx){qx=0.;onx=0.;}
        if(!(alpha<1.))break;
    }
    const double sum=warp_sum(q)+qx;if(sum>0.){const double inverse=1./sum;q*=inverse;qx*=inverse;}
    return true;
}
// The photon's recoils (engine.rs `kick`, `absorbed_recoil`): −k n and the absorbed light's -- the parts' shares (J.shares
// and J.beams, `count` of them; the explicit states' part J.direct last, its recoil's mean zero) moved to their projections'
// mean on the photon of ‖f‖² `size` (`part_weights`); `absorbed` picks the part, its leftover the explicit part's beam by J.drives,
// its lower bits the sign of its drives' mean's shot noise, and the rest of the exact conditional mean `mean` is added -- onto
// the momentum on the trap's principal axes.
__device__ void recoil(const Program& P,const JumpSpace& J,Traj& t,const double n[3],double wave_number,const double mean[3],int count,double size,double absorbed,int lane){
    double kick[3],mu[3]={0.,0.,0.},target[3],s=0.,on=0.,q,qx;
    if(lane<count){
        const Beam& b=P.beams[J.beams[lane]];const double k=(double)b.k;s=(double)J.shares[lane];on=(double)J.projections[lane]/size;
        #pragma unroll
        for(int a=0;a<3;a++)mu[a]=(double)b.direction[a]*k;
    }
    #pragma unroll
    for(int a=0;a<3;a++){kick[a]=-wave_number*n[a]+mean[a];target[a]=warp_sum(on*mu[a]);}
    if(part_weights(s,mu,(double)J.direct,target,count+1,(double)P.recoil_spread,q,qx)){
        double x=absorbed;int pick=-1;
        for(int c=0;c<count;c++){const double v=__shfl_sync(FULL,q,c);if(x<v){pick=c;break;}x-=v;}
        if(pick<0&&x<qx)pick=count;
        // Past the last part by round-off: the last with a weight.
        if(pick<0){if(qx>0.)pick=count;else for(int c=count-1;c>=0;c--)if(__shfl_sync(FULL,q,c)>0.){pick=c;break;}}
        #pragma unroll
        for(int a=0;a<3;a++)kick[a]-=warp_sum(q*mu[a]);
        if(pick<count){
            #pragma unroll
            for(int a=0;a<3;a++)kick[a]+=__shfl_sync(FULL,mu[a],pick);
        }else{
            const double left=fmin(fmax(x/qx,0.),1.);
            double total=0.;for(int i=0;i<J.ndrives;i++)total+=(double)J.drives[i];
            if(total>0.){
                double average[3]={0.,0.,0.};
                for(int i=0;i<J.ndrives;i++){const Beam& b=P.beams[J.driven[i]];const double w=(double)J.drives[i]*(double)b.k/total;for(int a=0;a<3;a++)average[a]+=w*(double)b.direction[a];}
                double y=left*total;int l=0;for(int i=J.ndrives-1;i>=0;i--)if(J.drives[i]>0.f){l=i;break;}
                for(int i=0;i<J.ndrives;i++){const double v=(double)J.drives[i];if(y<v){l=i;break;}y-=v;}
                const double bits=absorbed*67108864.,sign=bits-floor(bits)<0.5?1.:-1.;const Beam& b=P.beams[J.driven[l]];  // engine.rs `SIGN_BITS`
                #pragma unroll
                for(int a=0;a<3;a++)kick[a]+=(double)b.direction[a]*(double)b.k-average[a]+sign*average[a];
            }
        }
    }
    #pragma unroll
    for(int i=0;i<3;i++)t.p[i]+=P.axes[i][0]*kick[0]+P.axes[i][1]*kick[1]+P.axes[i][2]*kick[2];
}

__device__ __noinline__ void transient(const Program& P,Warp& W,Traj& t,const double u[5],Rng& rng,int lane,float2* E);
__device__ float2 response(const Program& P,const Warp& W,const Cluster& C,int c,int i,int col);
__device__ __noinline__ void jump(const Program& P,const Run& R,Warp& W,Traj& t,bool exact,int lane,float2* scratch){
    t.jumps++;Rng rng(R.seed,PURPOSE_JUMP,t.index,(u32)t.jumps);
    SPAN_BEGIN(pointing);point<false>(P,W,t,lane);SPAN_END(SPAN_JUMPPOINT,pointing);
    const Cluster C=P.clusters[t.cluster];
    {
        float s=0.f;for(int b=0;b<C.n;b++)s+=norm2(W.psi[b]);const float scale=1.f/sqrtf(s);
        __syncwarp();if(lane<C.n)W.psi[lane]=scale*W.psi[lane];__syncwarp();
    }
    SPAN_BEGIN(responses);
    if(exact)COUNT(COUNT_EXACT);
    const float2* X=exact&&exact_responses(P,W,C,lane,scratch,scratch+P.rows_at,scratch+P.between_at)?scratch:nullptr;
    SPAN_END(SPAN_EXACT,responses);SPAN_BEGIN(photons);
    if(C.fourth>=0)bent_state(P,W,C,t.time,lane);
    SPAN_BEGIN(matrices);float2* M=scratch+P.nnear*MAXN*3*NG;photon_matrices(P,W,C,lane,X,M,scratch+P.weights);SPAN_END(SPAN_MATRICES,matrices);SPAN_BEGIN(weightsp);
    JumpSpace& J=W.jump;
    // The class, and the cluster it lands the atom in: by Σ_a ‖u_a‖².
    double total=0.;
    for(int c=0;c<C.nclasses;c++){
        const Class K=P.classes[C.classes+c];const int nt=P.clusters[K.target].n0;
        float v=0.f;if(lane<3*nt)v=norm2(amplitude(P,W,C,K,lane%3,lane/3,t.time,M));
        const float w=warp_sum(v);if(lane==0)J.weights[c]=w;total+=(double)w;
    }
    __syncwarp();
    SPAN_END(SPAN_WEIGHTS,weightsp);SPAN_END(SPAN_PHOTONS,photons);
    if(!(total>0.)){t.threshold=rng.open_uniform();t.log_threshold=log(t.threshold);return;}
    SPAN_BEGIN(drawing);const Draws d=draw(rng,J,C.nclasses,total);SPAN_END(SPAN_DRAW,drawing);
    const Class K=P.classes[C.classes+d.pick];const Cluster T=P.clusters[K.target];const int nt=T.n0;
    // The transient's five draws, before the directions' loops.
    double tu[5];for(int i=0;i<5;i++)tu[i]=rng.uniform();
    if(lane<3*nt)J.u[lane%3][lane/3]=amplitude(P,W,C,K,lane%3,lane/3,t.time,M);
    __syncwarp();
    SPAN_BEGIN(emitting);double n[3],e[3];const float size=emit(rng,J,nt,d.polarization,lane,n,e);SPAN_END(SPAN_EMIT,emitting);
    // The recoils: the photon's and the absorbed light's about the exact conditional mean.
    SPAN_BEGIN(absorbed);
    float m[3];const int count=absorption(P,W,C,K,nt,e,t.time,X,lane,m);
    SPAN_END(SPAN_ABSORPTION,absorbed);
    SPAN_BEGIN(kicking);double mean[3];
    for(int x=0;x<3;x++)mean[x]=(double)m[x]/(double)size;
    recoil(P,J,t,n,K.wave_number,mean,count,(double)size,d.absorbed,lane);
    const float scale=1.f/sqrtf(size);
    __syncwarp();if(lane<MAXM)W.psi[lane]=lane<nt?scale*J.f[lane]:zero2();__syncwarp();
    t.cluster=K.target;t.logn=0.;t.threshold=d.threshold;t.log_threshold=log(d.threshold);
    SPAN_END(SPAN_KICK,kicking);SPAN_BEGIN(transiting);
    transient(P,W,t,tu,rng,lane,scratch+P.transient_at);SPAN_END(SPAN_TRANSIENT,transiting);
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
        const int at=C.positions+3*(lane+32*j),end=P.item_first[at+1];
        for(int i=P.item_first[at];i<end;i++){
            const int4 it=__ldg(P.items+i);
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

// The fewest products of B that sum exp(−iX) on a vector, ‖X‖ ≤ x, within `share` of ‖ψ‖ (engine.rs `series_within`): Taylor's
// series in equal parts, each through the first term whose bound y^m/m! is within its part's share. A part's round-off, about
// e^y of the single precision, stays within the share: y ≤ ln(share/ε).
struct Series{int parts,terms;};
__device__ Series series_within(float x,float share){
    const float widest=fmaxf(logf(share/(float)FLT_EPSILON),0.5f);Series best;int fewest=0x7fffffff;
    for(int parts=max((int)ceilf(x/widest),1);;parts++){
        const float y=x/(float)parts,part=share/(float)parts;int terms=1;float size=y;while(size>part&&terms<64){terms++;size*=y/(float)terms;}
        if(size>part)continue;
        if(parts*terms>=fewest)return best;
        fewest=parts*terms;best.parts=parts;best.terms=terms;
    }
}
// x ← exp(−iBτ)x for B's row-halves (engine.rs `Entries::cf4`, `series_body`), its series for the largest τ B's ‖·‖ allows.
__device__ float2 taylor(const float2 (&b)[8],const Series& S,float tau,float2 x,int lane,float4* v){
    const float step=tau/(float)S.parts;
    for(int p=0;p<S.parts;p++){
        float2 term=x,sum=x;
        for(int k=1;k<=S.terms;k++){float2 xb[8];half_of(term,lane,xb,v);const float2 z=times(b,xb);const float c=step/(float)k;term=make_float2(z.y*c,-z.x*c);sum+=term;}
        x=sum;
    }
    return x;
}

// A cluster's segment of an exact step (engine.rs `Segment`): its size and parts, the parts it keeps (bit
// q; bit 0 the static one), its grid's spacing from the step's start, its path -- beating (CF4 over each
// spacing) or static (the series on the vector) -- and its exponentials' series, and
// in the scratch its parts' phases at the step's start, their norms (`exact_norms`) and their Ω_q. Its
// times are single precision from the step's start (a step's τ holds them to 1e-7 of itself).
struct Segment{int n,parts;unsigned kept;int beating;double start,spacing;float delta;Series series;const float2* base;const float* F;const float* w;float4* v;};

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
    for(int q=1;q<S.parts;q++){
        if(!((S.kept>>q)&1u))continue;
        const float2 p1=beat_phase(S,q,t1),p2=beat_phase(S,q,t2),z1=w1*p1+w2*p2,z2=w2*p1+w1*p2;
        float2 a[8];row_half(X+q*nn,S.n,lane,a);
        #pragma unroll
        for(int j=0;j<8;j++){b1[j]+=z1*a[j];b2[j]+=z2*a[j];}
    }
    x=taylor(b1,S.series,len,x,lane,S.v);
    return taylor(b2,S.series,len,x,lane,S.v);
}

// x over [t, t + len) by a segment (engine.rs `Segment::advance`): CF4 where it beats; else the series of
// exp(−iH₀ len) on the vector -- the CPU's propagator over a whole spacing, as exact: a single-precision
// propagator applied at every spacing would compound its own rounding over the step (1e-4 of ln‖ψ‖² over 300
// spacings), the series' rounding is new at each.
__device__ float2 seg_advance(const Segment& S,const float2* X,float t,float len,float2 x,int lane){
    if(S.beating)return cf4(S,X,t,len,x,lane);
    float2 a[8];row_half(X,S.n,lane,a);return taylor(a,S.series,len,x,lane,S.v);
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
    for(int q=1;q<S.parts;q++){
        if(!((S.kept>>q)&1u))continue;
        const float2 p1=beat_phase(S,q,ta),p2=beat_phase(S,q,tb),p3=beat_phase(S,q,ta+len),p4=beat_phase(S,q,tb+len);
        const float2 za1=w1*p1+w2*p2,za2=w2*p1+w1*p2,zb1=w1*p3+w2*p4,zb2=w2*p3+w1*p4;
        float2 h[8];row_half(X+q*nn,S.n,lane,h);
        #pragma unroll
        for(int j=0;j<8;j++){a1[j]+=za1*h[j];a2[j]+=za2*h[j];b1[j]+=zb1*h[j];b2[j]+=zb2*h[j];}
    }
    x=taylor(a1,S.series,len,x,lane,S.v);middle=taylor(a2,S.series,len,x,lane,S.v);
    x=taylor(b1,S.series,len,middle,lane,S.v);last=taylor(b2,S.series,len,x,lane,S.v);
}
// A panel's two halves from x (`seg_advance` twice): a beating segment's by `cf4_panel`, a static one's with H₀ loaded once.
__device__ void seg_advances(const Segment& S,const float2* X,float t,float half,float2 x,int lane,float2& middle,float2& last){
    if(S.beating){cf4_panel(S,X,t,half,x,lane,middle,last);return;}
    float2 a[8];row_half(X,S.n,lane,a);middle=taylor(a,S.series,half,x,lane,S.v);last=taylor(a,S.series,half,middle,lane,S.v);
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
// p, or its adjoint (model.rs `components`' `scatter`): ee S into M (H_p until the end), on Q into Γ_p -- ee and the first order's --,
// −(c S + ee Σ_f ∇s_f ∂_f S) into G_p per axis and, of a direct term of a pair with a first order (its `Group::first` in `firsts`), its gradient's coefficients on Q.
__device__ __forceinline__ void group_term(const Program& P,const Warp& W,const Cluster& C,const float2* s,const float2* q,float2 ee,float2 on,const float2 (&c)[3],int first,const FirstPair* firsts,
        int p,bool adjoint,int e,float2* X,float2* G,float2* Gamma,int nn){
    const Point& pt=W.pt;float2 ds[MOST_FOCUSED];const float2 sv=entry_slopes(P,W,s,C.size,ds),qv=entry(P,W,q,C.size,-1);
    float2 h=ee*sv,g=on*qv;if(adjoint){h=conj(h);g=conj(g);}
    X[p*nn+e]+=h;Gamma[p*nn+e]+=g;
    #pragma unroll
    for(int a=0;a<3;a++){
        float2 v=zero2()-c[a]*sv;
        #pragma unroll
        for(int f=0;f<MOST_FOCUSED;f++)if(f<P.focused)v-=(pt.grad_s[f][a]*ee)*ds[f];
        if(adjoint)v=conj(v);else if(first>0)v+=firsts[first-1].grad[a]*qv;
        G[(3*p+a)*nn+e]+=v;
    }
}
__device__ __forceinline__ void atomic_add(float2* x,float2 v){atomicAdd(&x->x,v.x);atomicAdd(&x->y,v.y);}

// A cluster's components at the point (model.rs `Model::components`) into X: per part q, H_q -- its
// groups' −S, at part 0 h0 less the scalar, −(i/2) its groups' Q with the first order's, what its transfer leaves, its explicit states' terms -- and G_q,
// then the gradients' Hermitian parts (G₀'s; per pair ±Ω, (G_Ω + G_{−Ω}†)/2 at Ω and its adjoint at −Ω).
// The lanes take the entries e = lane + 32j; a pair standing for two reads its adjoint at the mirror's place, as a group between two carriers
// its first order's, at the mirror's part (of the part p, Q − p: the beats ascend and close under negation). Its first orders' table in X after the segment's room.
__device__ __noinline__ void exact_components(const Program& P,Warp& W,const Cluster& C,int lane,float2* X){
    const int n=C.n,nn=n*n,Q=C.parts;const float2* tables=P.tables+C.table;const Point& pt=W.pt;
    float2* G=X+Q*nn;float2* Gamma=X+4*Q*nn;FirstPair* firsts=(FirstPair*)(X+(5*Q+4)*nn+4*Q);
    for(int e=lane;e<5*Q*nn;e+=32)X[e]=zero2();
    first_table(P,W,C,0.,false,lane,firsts);
    __syncwarp();
    for(int gi=0;gi<C.ngroups;gi++){
        const Group Gr=P.groups[C.groups+gi];if(Gr.count!=nn)continue;
        const int p=P.group_parts[C.group_parts+gi],beat=p==0?0:Q-p;const Fields& fa=pt.fields[Gr.absorbed];const Fields& fe=pt.fields[Gr.emitted];const bool mirrored=Gr.first>0&&!Gr.hermitian;
        for(int pi=0;pi<Gr.npairs;pi++){
            const int2 pair=P.pairs[Gr.pairs+pi];const int k1=pair.x,k=pair.y;const bool twice=Gr.hermitian&&k1<k;
            const float2 ee=conj(fe.e[k1])*fa.e[k];float2 c[3];
            #pragma unroll
            for(int a=0;a<3;a++)c[a]=conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a];
            const int first=Gr.first>0?Gr.first+pi:0;const float2 on=first>0?firsts[first-1].on:zero2(),off=first>0?firsts[first-1].off:zero2();
            const float2* s=tables+Gr.s+pi*nn;const float2* q=tables+Gr.q+pi*nn;
            for(int e=lane;e<nn;e+=32){
                const int r=e/n,m=(e-r*n)*n+r;
                group_term(P,W,C,s+e,q+e,ee,ee+on,c,first,firsts,p,false,e,X,G,Gamma,nn);
                if(twice)group_term(P,W,C,s+m,q+m,ee,ee+off,c,0,firsts,p,true,e,X,G,Gamma,nn);
                else if(mirrored)Gamma[beat*nn+e]+=conj(off*entry(P,W,q+m,C.size,-1));
            }
        }
    }
    // The sparse groups' entries at their places (`items`) and what the transfer leaves of any group; their mirrors', an adjoint's from its pair at the mirror.
    if(C.nbeats>0)for(int e=lane;e<nn;e+=32){
        const int at=C.positions+3*e,kept=P.item_first[at+1],mirrors=P.item_first[at+2],end=P.item_first[at+3];
        for(int i=P.item_first[at];i<kept;i++){
            const int4 it=__ldg(P.items+i);const int k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
            const int p=P.residual_parts[C.residual_parts+(it.z>>12)],first=it.w>>ITEM_PAIR;const float2 ee=conj(fe.e[k1])*fa.e[k];float2 c[3];
            #pragma unroll
            for(int a=0;a<3;a++)c[a]=conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a];
            group_term(P,W,C,tables+it.x,tables+it.y,ee,first>0?ee+firsts[first-1].on:ee,c,first,firsts,p,false,e,X,G,Gamma,nn);
        }
        for(int i=kept;i<mirrors;i++){
            const int4 it=__ldg(P.items+i);const int k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
            const int p=P.residual_parts[C.residual_parts+(it.z>>12)];float2 ws,wq,mu,nu;entry_pair(P,pt,tables+it.x,tables+it.y,C.size,ws,wq);first_coefficients(ws,wq,mu,nu);
            Gamma[p*nn+e]+=mu*(conj(fe.e[k1])*fa.doppler[k]);
            #pragma unroll
            for(int a=0;a<3;a++)G[(3*p+a)*nn+e]+=ws*xi_slope(fe,fa,k1,k,a);
        }
        for(int i=mirrors;i<end;i++){
            const int4 it=__ldg(P.items+i);const int role=it.w&ITEM_ROLE,first=it.w>>ITEM_PAIR,k1=it.z&3,k=(it.z>>2)&3;const Fields& fa=pt.fields[(it.z>>4)&7];const Fields& fe=pt.fields[(it.z>>7)&7];
            const int p=P.residual_parts[C.residual_parts+(it.z>>12)],beat=p==0?0:Q-p;
            if(role==ITEM_ADJOINT){
                const float2 ee=conj(fe.e[k1])*fa.e[k];float2 c[3];
                #pragma unroll
                for(int a=0;a<3;a++)c[a]=conj(fe.grad[k1][a])*fa.e[k]+conj(fe.e[k1])*fa.grad[k][a];
                group_term(P,W,C,tables+it.x,tables+it.y,ee,first>0?ee+firsts[first-1].off:ee,c,0,firsts,p,true,e,X,G,Gamma,nn);
            }else if(role==ITEM_FIRST)Gamma[beat*nn+e]+=conj(firsts[first-1].off*entry(P,W,tables+it.y,C.size,-1));
            else{float2 ws,wq,mu,nu;entry_pair(P,pt,tables+it.x,tables+it.y,C.size,ws,wq);first_coefficients(ws,wq,mu,nu);Gamma[beat*nn+e]+=nu*conj(conj(fe.e[k1])*fa.doppler[k]);}
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
    // The fourth order (a cluster of ground states): each slab's kept entries in their beats' parts (model.rs `fourth_parts`; the
    // force keeps the second order's).
    if(C.fourth>=0){
        __syncwarp();const Fourth F=P.fourths[C.fourth];
        for(int q=0;q<F.nquartic;q++){
            const Quartic Q=P.quartics[F.quartic+q];const float2 m=monomial(pt,Q);if(zero(m))continue;
            for(int e=lane;e<nn;e+=32){const int part=P.quartic_parts[Q.parts+e];if(part>=0)X[part*nn+e]+=m*entry(P,W,tables+Q.at+e,C.size,-1);}
        }
    }
    // The explicit states' terms (model.rs `exact_terms`), a lane each: a kept drive's Σ_k E_k U_k, a decay
    // cross term's (i/2) Σ_k (E_k X_k − E^D_k X2_k), a decay's value, a dressing's drive's Σ_k E_k Y_k, and their mirrors at the
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
            float2 z=zero2();for(int k=0;k<rank;k++){float2 x,x2;entry_pair(P,pt,tables+T.at+k,tables+T.at2+k,C.size,x,x2);z+=fl.e[k]*x-fl.doppler[k]*x2;}
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
// horizon --, the cheapest first while their sums stay within the share (`Settings::beats`); a side whose first boundary
// term alone exceeds the share takes no commutators, that term its cost. The parts
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
            if(2.*m/w*once>P.beats){x+=2.*m/w*once;continue;}
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
// into `beats`, the trajectory's largest), and its grid and series: with beats kept, whole panels of two spacings whose phase
// (‖H₀‖₁ + Σ‖H_j‖₁ + max|Ω|/2)δ is within the grid's (`Program::grid_phase`), each exponential within its part of the larger of
// the truncation and the beats' share and at least of single precision's round-off, ‖w₁H(t₁) + w₂H(t₂)‖ ≤ ½‖H₀‖₁ + (|w₁| + |w₂|)Σ‖H_j‖₁;
// else ‖H₀‖₁δ within the node phase, the series to the round-off, and its decay i(H₀ − H₀†) for the roots.
__device__ __noinline__ Segment exact_segment(const Program& P,Warp& W,int k,double start,double tau,int lane,float2* X,float& beats){
    const Cluster C=P.clusters[k];Segment S;S.n=C.n;S.parts=C.parts;S.start=start;
    const int n=C.n,nn=n*n,Q=C.parts;const double* omegas=P.omegas+C.omegas;
    float2* base=X+(5*Q+4)*nn;float* F=(float*)(base+Q);float* w=F+5*Q;S.base=base;S.F=F;S.w=w;
    SPAN_BEGIN(componentsp);exact_components(P,W,C,lane,X);SPAN_END(SPAN_COMPONENTS,componentsp);
    for(int m=0;m<4*Q;m++){float column;const float s=msize(X+m*nn,n,lane,column);if(m<Q){F[m]=column;F[Q+m]=s;}else F[2*Q+m-Q]=s;}
    if(lane<Q){base[lane]=lane==0?make_float2(1.f,0.f):cis(omegas[lane-1]*start);w[lane]=lane==0?0.f:(float)omegas[lane-1];}
    __syncwarp();
    double dropped;S.kept=exact_resolve(P,X,F,omegas,n,Q,tau,lane,dropped);beats=fmaxf(beats,(float)(dropped/tau));
    S.beating=(S.kept&~1u)!=0u;
    if(!S.beating){
        const int steps=max((int)ceil((double)F[0]*tau/P.node_phase),1);S.spacing=tau/(double)steps;
        S.series=series_within(F[0]*(float)S.spacing,EXACT_ROUND);
        float2* D=X+(5*Q+1)*nn;
        for(int e=lane;e<nn;e+=32){const int r=e/n,c=e-r*n;const float2 z=X[e]-conj(X[c*n+r]);D[e]=make_float2(-z.y,z.x);}
        __syncwarp();
    }else{
        double beat=0.,most=0.;
        for(int q=1;q<Q;q++)if((S.kept>>q)&1u){beat+=(double)F[q];most=fmax(most,fabs(omegas[q-1]));}
        const double widest=P.grid_phase/((double)F[0]+beat+0.5*most);const int steps=2*max((int)ceil(0.5*tau/widest),1);S.spacing=tau/(double)steps;
        S.series=series_within((float)((0.5*(double)F[0]+0.57735026918962576*beat)*S.spacing),fmaxf((float)(fmax((double)P.truncation,P.beats)/(double)(2*steps)),EXACT_ROUND));
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
        const float2 y=seg_advance(S,X,t+from,s-from,first?x:middle,lane);
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
__device__ float exact_evolve(const Program& P,Warp& W,const Segment& S,const float2* X,float2& x,float at,float remaining,float target,int lane,bool& jumped,float impulse[3]){
    const float end=at+remaining,delta=S.delta;float t=at;
    int i=max((int)ceilf(at/delta),0);
    float fa[3];float na=seg_force(S,X,t,x,lane,fa);jumped=false;
    while(t<end){
        const float node=(float)i*delta;float b;bool whole=false;
        if(node>t)b=fminf(node,end);else{const float b2=(float)(i+2)*delta;if(b2<=end){b=b2;whole=true;}else b=end;}
        const float width=b-t,half=0.5f*width;
        float2 middle,last;seg_advances(S,X,t,half,x,lane,middle,last);
        SPAN_BEGIN(forcesp);float fm[3],fb[3],nm,nb;seg_forces(S,X,t+half,b,middle,last,lane,fm,fb,nm,nb);SPAN_END(SPAN_FORCES,forcesp);
        COUNT_BY(COUNT_NODES,2);COUNT_BY(COUNT_PRODUCTS,2*(S.beating?2:1)*S.series.parts*S.series.terms);
        if(logf(nb)<=target){
            const float s=exact_root(S,X,x,middle,t,na,nm,nb,width,target,lane);
            const float2 h=seg_advance(S,X,t,0.5f*s,x,lane),y=seg_advance(S,X,t+0.5f*s,0.5f*s,h,lane);
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

// The exact step of τ (engine.rs `Engine::exact`): the flow for τ/2 at the step's reference H (`reference`, at its start); at the point there every cluster the
// trajectory visits takes its segment once (`exact_segment`), ψ through the step with the jumps at their
// roots and the impulse along it; the kick, less τ w times the harmonic force about the centre at the point; the flow for τ/2 at H.
__device__ __noinline__ void exact_step(const Program& P,const Run& R,Warp& W,Traj& t,double tau,bool exact,int lane,float2* scratch){
    COUNT(COUNT_ADVANCES);
    const Reference H=reference(P,t.q);
    flow(P,t,0.5*tau,H);point<true>(P,W,t,lane);
    const double start=t.time,ph[3]={t.p[0],t.p[1],t.p[2]};float2* X=scratch+P.exact_at;
    double impulse[3]={0.,0.,0.},elapsed=0.;int built=-1;Segment S;
    __syncwarp();float2 x=W.psi[lane&15];
    while(elapsed<tau){
        if(built!=t.cluster){
            // A later cluster's segment at the step's point (the jump's moved W.pt to its own momentum).
            if(built>=0){Traj h=t;for(int i=0;i<3;i++)h.p[i]=ph[i];point<true>(P,W,h,lane);}
            SPAN_BEGIN(segment);S=exact_segment(P,W,t.cluster,start,tau,lane,X,t.beats);S.v=W.vec;built=t.cluster;SPAN_END(SPAN_GENERATOR,segment);
        }
        bool jumped;float part[3]={0.f,0.f,0.f};
        SPAN_BEGIN(evolving);const float taken=exact_evolve(P,W,S,X,x,(float)elapsed,(float)(tau-elapsed),(float)(t.log_threshold-t.logn),lane,jumped,part);SPAN_END(SPAN_EVOLVE,evolving);
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
    for(int i=0;i<3;i++)t.p[i]+=P.axes[i][0]*impulse[0]+P.axes[i][1]*impulse[1]+P.axes[i][2]*impulse[2]+tau*H.w*P.stiffness[i]*(t.q[i]-H.c[i]);
    flow(P,t,0.5*tau,H);t.time=start+tau;
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
                const int at=C.positions+3*e,end=P.item_first[at+1];
                for(int i=P.item_first[at];i<end;i++){
                    const int4 it=__ldg(P.items+i);
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

// A table entry of a carrier's response T_k at the stencil (model.rs `responses`' tables: rank × NE × n0).
__device__ __forceinline__ const float2* t_table(const Program& P,const Cluster& C,int c,int k,int i,int col){
    return P.tables+C.table+P.carrier_tables[C.carriers+3*c+2]+(k*NE+i)*C.n0+col;
}

// A bound of the ringing's probability p = Σ_k ‖e_k‖² from the columns' norms (model.rs `ringing_bound`), Σ_k (Σ_channels |ψ′_b| Σ_r |E_r| ‖T_r[:,b]‖)²: the
// norms lie right after T (rank × n0, real). A lane per channel of each coherent group.
__device__ float ringing_bound(const Program& P,const Warp& W,const Cluster& C,int lane){
    float bound=0.f;
    for(int k=0;k<C.ncoherent;k++){
        const Coherent G=P.coherent[C.coherent+k];float s=0.f;
        for(int h=lane;h<G.count;h+=32){
            const Channel ch=P.channels[G.channels+h];const int rank=P.carriers[ch.carrier].rank;const float a=sqrtf(norm2(W.psi[ch.col]));
            const float2* x=P.tables+C.table+P.carrier_tables[C.carriers+3*ch.carrier+2]+rank*NE*C.n0+ch.col;
            for(int r=0;r<rank;r++)s+=a*sqrtf(norm2(W.pt.fields[ch.carrier].e[r]))*entry(P,W,x+r*C.n0,C.size,-1).x;
        }
        s=warp_sum(s);bound+=s*s;
    }
    return bound;
}

// The ringing's photon (engine.rs `ringing_photon`; model.rs `ringing`, `ringing_photon`, `transient_photon`) from ψ′ (W.psi), E its amplitudes e_k per coherent group, the
// photon's own v_k = −e_k: the group k and the manifold m by their weights (x within their sum), the time τ ~ Γ e^{−Γτ} at the manifold's width (u[1]), the amplitude
// u_a[to] = Σ_i O_a[to, i] v_k[i], O_a = C_a e^{−iHτ} on the manifold, each landing state at its frame at t + τ, relative to its cluster's first state, its landing
// cluster by the weights there (u[2]), its direction and polarization (`emit`, u[3]), its kicks a jump's (`recoil`, u[4]).
// E's room goes on past the groups' amplitudes (program.rs `Program::new`): O; per carrier and rank the channels' Y_r[i] = Σ_channels e^{−iε̃t} ψ′_b T_r[i, b]; H's off-diagonal
// part, then its propagator e^{−iHτ}; its eigenvectors; its diagonal, in double precision, then its eigenvalues; their phases.
__device__ __noinline__ void ringing_photon(const Program& P,Warp& W,Traj& t,double x,const double u[5],Rng& rng,int lane,float2* E){
    const Cluster C=P.clusters[t.cluster];const Point& pt=W.pt;JumpSpace& J=W.jump;const double time=t.time;
    int k=0,m=0;bool found=false;
    for(int g=0;g<C.ncoherent&&!found;g++)for(int l=0;l<P.nlevels&&!found;l++){
        const int first=P.level_first[l],count=P.level_first[l+1]-first;
        const double w=(double)warp_sum(lane<count?norm2(E[g*NE+P.levels[first+lane]]):0.f);
        if(w>0.){k=g;m=l;if(x<w)found=true;else x-=w;}
    }
    const Coherent G=P.coherent[C.coherent+k];const int M=P.level_first[m+1]-P.level_first[m];const int* L=P.levels+P.level_first[m];const float2* e=E+k*NE;
    float2* O=E+C.ncoherent*NE;float2* Y=O+3*NG*M;float2* Z=Y+P.ncarriers*MOST_RANK*M;float2* V=Z+M*M;double* D=(double*)(V+M*M);float2* phases=(float2*)(D+M);
    const double width=P.level_width[m],tau=-log(1.-u[1])/width;
    // H's block on the manifold, decoupled from the cluster's explicit states as its folded responses are (ground.rs `ringing_block`): the decay out of K's off-diagonal
    // entries, its diagonal's mean apart before single precision takes it (its phase comes back in O). Its eigenstates as far as a coupling matters over the width.
    const double own=lane<M?diagonal_at(P,pt,NG+L[lane]):0.,mean=warp_sum(own)/(double)M;
    if(lane<M)D[lane]=own-mean;
    const u32 explicit_states=P.explicit_of[C.states[0]];
    for(int a=lane;a<M*M;a+=32){
        const int i=a/M,j=a-i*M,p=L[min(i,j)],q=L[max(i,j)];float2 h=zero2();
        if(i!=j&&kept_entry(explicit_states,p,q)){const float2 d=P.decay[p*NE+q];h=excited_at(P,pt,p,q)+make_float2(-0.5f*d.y,0.5f*d.x);if(i>j)h=conj(h);}
        Z[a]=h;
    }
    __syncwarp();
    hermitian_eigen(Z,V,D,M,0.f,P.ringing,(float)width,lane);
    // Its propagator over τ, V e^{−iEτ} V†, into Z.
    if(lane<M)phases[lane]=cis(-D[lane]*tau);
    __syncwarp();
    for(int a=lane;a<M*M;a+=32){
        const int i=a/M,j=a-i*M;float2 s=zero2();
        for(int l=0;l<M;l++)s+=V[i*M+l]*phases[l]*conj(V[j*M+l]);
        Z[a]=s;
    }
    __syncwarp();
    // O_a[to, i] at lane 3·to + a, and there the photon's amplitude u_a[to].
    const int to=lane/3,ax=lane-3*to;float2 v=zero2();
    if(lane<3*NG){
        const int frames=P.clusters[P.ground_cluster[to]].frames;const float2 frame=cis((P.frames[frames+P.ground_position[to]]-P.frames[frames])*(time+tau)-mean*tau);
        float2* out=O+lane*M;
        for(int i=0;i<M;i++)out[i]=zero2();
        for(int j=0;j<M;j++){const float2 c=P.emission[(ax*NG+to)*NE+L[j]];if(zero(c))continue;const float2 z=frame*c;for(int i=0;i<M;i++)out[i]+=z*Z[j*M+i];}
        for(int i=0;i<M;i++)v-=out[i]*e[L[i]];
    }
    // Its landing cluster by the weights there.
    double total=0.;
    for(int c=0;c<P.nclusters;c++){const float w=warp_sum(lane<3*NG&&P.ground_cluster[to]==c?norm2(v):0.f);if(lane==0)J.weights[c]=w;total+=(double)w;}
    __syncwarp();
    if(!(total>0.))return;
    const int target=picked(J,P.nclusters,u[2]*total),nt=P.clusters[target].n0;const bool row=lane<NG&&P.ground_cluster[lane]==target;
    if(lane<3*MAXN)J.u[lane%3][lane/3]=zero2();
    __syncwarp();
    if(lane<3*NG&&P.ground_cluster[to]==target)J.u[ax][P.ground_position[to]]=v;
    __syncwarp();
    double n[3],polar[3];const float size=emit(rng,J,nt,u[3],lane,n,polar);
    // O along the polarization, c[to, i] = Σ_a e_a O_a[to, i], in its first axis's rows at lane `to`.
    if(row)for(int i=0;i<M;i++){float2 s=zero2();for(int a=0;a<3;a++)s+=(float)polar[a]*O[(3*lane+a)*M+i];O[3*lane*M+i]=s;}
    for(int a=lane;a<P.ncarriers*MOST_RANK*M;a+=32)Y[a]=zero2();
    __syncwarp();
    // A lane per state i of the manifold: its Y_r[i], and the amplitude's gradient along the polarization on the photon, Σ_to f*[to] c[to, i] Σ_channels e^{−iε̃t} ψ′_b Σ_r
    // (∇E_r T_r + E_r Σ_f ∂_f T_r ∇s_f)[i, b], the exact conditional mean.
    float along[3]={0.f,0.f,0.f};
    if(lane<M){
        const int i=L[lane];float2 g=zero2();
        for(int b=0;b<NG;b++)if(P.ground_cluster[b]==target)g+=conj(J.f[P.ground_position[b]])*O[3*b*M+lane];
        for(int h=0;h<G.count;h++){
            const Channel ch=P.channels[G.channels+h];const Fields& fl=pt.fields[ch.carrier];const float2 a=residual(-ch.residual,time)*W.psi[ch.col],weight=g*a;
            for(int r=0;r<P.carriers[ch.carrier].rank;r++){
                float2 ds[MOST_FOCUSED];const float2 tv=entry_slopes(P,W,t_table(P,C,ch.carrier,r,i,ch.col),C.size,ds);
                Y[(ch.carrier*MOST_RANK+r)*M+lane]+=a*tv;
                #pragma unroll
                for(int d=0;d<3;d++){
                    float2 s=fl.grad[r][d]*tv;
                    #pragma unroll
                    for(int f=0;f<MOST_FOCUSED;f++)if(f<P.focused)s+=pt.grad_s[f][d]*(fl.e[r]*ds[f]);
                    along[d]+=(weight*s).y;
                }
            }
        }
    }
    __syncwarp();
    // Of the photon's own sign: this gradient's amplitude is Σ e^{−iε̃t} R ψ′, the photon J.u its negative.
    double exact_mean[3];for(int d=0;d<3;d++)exact_mean[d]=-(double)warp_sum(along[d])/(double)size;
    // Per beam of the group's carriers, ascending and each one's beams in order, its part a_l Σ_i c[to, i] Σ_r c_lr Y_r[i], of the gradient's sign, the photon's negative: its share and its projection on the photon.
    int count=0;
    for(int c=0;c<P.ncarriers;c++){
        bool in=false;for(int h=0;h<G.count;h++)if(P.channels[G.channels+h].carrier==c)in=true;
        if(!in)continue;const Carrier Rc=P.carriers[c];
        for(int b=0;b<Rc.beams;b++){
            const int l=P.beam_list[Rc.first+b];const Beam& B=P.beams[l];float2 part=zero2(),fr=zero2();
            if(row){
                for(int i=0;i<M;i++){float2 z=zero2();for(int r=0;r<Rc.rank;r++)z+=make_float2(B.coefficients[r][0],B.coefficients[r][1])*Y[(c*MOST_RANK+r)*M+i];part+=O[3*lane*M+i]*z;}
                part=pt.a[l]*part;fr=J.f[P.ground_position[lane]];
            }
            const float w=warp_sum(norm2(part)),w1=-warp_sum(fr.x*part.x+fr.y*part.y);if(lane==0){J.shares[count]=w;J.projections[count]=w1;J.beams[count]=l;}count++;
        }
    }
    __syncwarp();
    if(lane==0){J.direct=0.f;J.ndrives=0;}
    __syncwarp();
    // The photon's wave number: its group's first channel's carrier's.
    recoil(P,J,t,n,P.carriers[P.channels[G.channels].carrier].wave,exact_mean,count,(double)size,u[4],lane);
    const float scale=1.f/sqrtf(size);
    __syncwarp();if(lane<MAXM)W.psi[lane]=lane<nt?scale*J.f[lane]:zero2();__syncwarp();
    t.cluster=target;
}

// The post-jump transient (engine.rs, # The transient; `transient`) from the normalized ground state ψ′ the jump left (W.psi, its n0 ground states) with the jump's five draws u: with
// probability p = Σ_k ‖e_k‖², e_k = Σ_channels e^{−iε̃t} R_c ψ′ per coherent group (X_k ψ′ = −e_k; model.rs `ringing_amplitudes`), the ringing's photon (`ringing_photon`) and the slow part
// of the state it leaves (`slow_start`); else at fourth order the slow part Πψ′/√A with the threshold θ (1 − p)/A, A = ‖Πψ′‖², p the metric's form on ψ′, which Π holds. Without the fourth
// order's tables the atom stays in ψ′, and a jump asks for no amplitude unless its draw falls within a bound of p (`ringing_bound`). E: the scratch's room for e_k, lane i its row i, and for the photon.
__device__ __noinline__ void transient(const Program& P,Warp& W,Traj& t,const double u[5],Rng& rng,int lane,float2* E){
    const Cluster C=P.clusters[t.cluster];float form=0.f;
    if(C.fourth>=0){
        const Entries entries(C.n,lane);float2 x[SLOTS],pi[SLOTS];
        slow_and_metric(P,W,C,entries,t.time,lane,x,pi);form=quadratic(W,entries,x,lane);
        if(u[0]>=(double)form){
            const float a=slow_part(W,entries,pi,lane);
            t.threshold*=(1.-(double)form)/(double)a;t.log_threshold=log(t.threshold);return;
        }
    }else if(u[0]>=(double)ringing_bound(P,W,C,lane))return;
    float total=0.f;
    for(int k=0;k<C.ncoherent;k++){
        const Coherent G=P.coherent[C.coherent+k];float2 e=zero2();
        if(lane<NE){
            for(int h=0;h<G.count;h++){const Channel ch=P.channels[G.channels+h];e+=residual(-ch.residual,t.time)*response(P,W,C,ch.carrier,lane,ch.col)*W.psi[ch.col];}
            E[k*NE+lane]=e;total+=norm2(e);
        }
    }
    total=warp_sum(total);__syncwarp();
    // At fourth order the draw is within the metric's form, X on the fourth order's carriers; without it within the bound, and the amplitudes' own weight decides.
    const double x=form>0.f?u[0]*(double)total/(double)form:u[0];
    if(!(x<(double)total))return;
    ringing_photon(P,W,t,x,u,rng,lane,E);
    slow_start(P,W,t,lane);
}

// One step of dt (engine.rs `Engine::step`): flow, kick and flow; where ‖ψ‖² crosses the threshold
// the step is redone to the crossing, the jump made there and the rest taken from the state it left.
// True when the atom is lost in it: unbound -- its mechanical energy above the trap's far level --
// outside the trap's region, judged at the step's middle where it went without a jump, else at its
// end.
__device__ bool step(const Program& P,const Run& R,Warp& W,Traj& t,int lane,float2* scratch){
    // The first order in the Doppler shift where it holds, by the step's start; the exact response beyond: read at its jumps and by an exact step alone.
    double left=P.dt;Local L;Middle M;bool fresh=false,exact=false,asked=false;
    if(P.clusters[t.cluster].exact){exact=doppler_ratio(P,t,lane)>P.allowance;exact_step(P,R,W,t,left,exact,lane,scratch);point<false>(P,W,t,lane);fresh=true;}
    else for(;;){
        const Traj start=t;
        if(lane<MAXN)W.start[lane]=W.psi[lane];
        double lh;advance(P,W,t,left,lane,lh,L,M);
        if(t.logn>t.log_threshold)break;
        // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
        if(!asked){exact=doppler_ratio(P,start,lane)>P.allowance;asked=true;}
        double ta,tb,la,lb;
        if(lh<=t.log_threshold){ta=0.;tb=0.5*left;la=start.logn;lb=lh;}else{ta=0.5*left;tb=left;la=lh;lb=t.logn;}
        const double at=lb<la?fmin(fmax(ta+(t.log_threshold-la)/(lb-la)*(tb-ta),ta),tb):tb;
        t=start;__syncwarp();if(lane<MAXN)W.psi[lane]=W.start[lane];__syncwarp();
        if(at>0.){double unused;advance(P,W,t,at,lane,unused,L,M);}
        {COUNT(COUNT_JUMPS);SPAN_BEGIN(jumping);jump(P,R,W,t,exact,lane,scratch);SPAN_END(SPAN_JUMP,jumping);}
        left-=at;
        if(!(left>0.)){point<false>(P,W,t,lane);fresh=true;break;}
        if(P.clusters[t.cluster].exact){exact_step(P,R,W,t,left,exact,lane,scratch);point<false>(P,W,t,lane);fresh=true;break;}
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
    point<false>(P,W,t,lane);
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
// The kernel: every slot of the wave with a trajectory (each its warp and block) from where it stands to the output
// interval `end`, for its cycle budget. Interval 0 is the initial state; interval i > 0 takes the steps
// ((i − 1)·steps, i·steps], the last ones past its task's last output its reach (solve.rs # Batches: a group runs only as far as the outputs that want it). Each warp runs its trajectory on its own,
// keeping its record of each interval it finishes (the output's point record, the windows closing in it) in its slot's ring
// of records, and leaves after the step that spends its budget -- a warp of cheap steps does not wait for one of dear ones
// (an exact step costs ten of a folded one) --, its place in its state and the interval it stands in in `costs`. `task_sums` sums
// the intervals every slot has passed.
extern "C" __global__ void __launch_bounds__(32,16) motion3d(const Program P,const Run R){
    __shared__ Warp W;
    const int lane=threadIdx.x,slot=blockIdx.x;if(R.tasks[3*(slot/WARPS)+1]<=slot%WARPS)return;
    State* S=R.states+slot;
    const int through=R.tasks[3*(slot/WARPS)],intervals=through+1+(R.reach+R.steps-1)/R.steps,until=min(R.end,intervals);
    if(S->interval>=until){if(lane==0){R.costs[3*slot]=0;R.costs[3*slot+1]=0;R.costs[3*slot+2]=S->interval>=intervals?PAST_ALL:(unsigned long long)S->interval;}return;}
    double* ring=R.rings+(size_t)slot*R.ring_stride;float2* scratch=R.scratch+(size_t)slot*R.scratch_stride;
    double* records=R.records+(size_t)slot*MOST_INTERVALS*SLOT_RECORD;
    Traj t;Twin h;double prefix=0.,q0[3],p0[3];const bool held=R.held.u!=nullptr;
    long long twin_cycles=0;const long long launched=clock64();
    {
        #pragma unroll
        for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];q0[i]=S->q0[i];p0[i]=S->p0[i];}
        t.time=S->time;t.logn=S->logn;t.threshold=S->threshold;t.log_threshold=S->log_threshold;
        t.cluster=S->cluster;t.jumps=S->jumps;t.death=S->death;t.index=S->index;t.beats=S->beats;
        h.logn=S->twin_logn;h.threshold=S->twin_threshold;h.log_threshold=S->twin_log_threshold;h.cluster=S->twin_cluster;h.jumps=S->twin_jumps;
        if(lane<15)prefix=S->prefix[lane];
        if(lane<MAXM)W.psi[lane]=S->psi[lane];
        if(lane<MAXN)W.twin[lane]=S->twin[lane];
    }
#ifdef PROFILE
    if(lane<SPANS)W.profile[lane]=0;
#endif
    __syncwarp();SPAN_BEGIN(launch);
#ifdef PROFILE
    if(lane==0){W.profile[SPAN_CHECKPOINT]+=launch-launched;W.profile[COUNT_LAUNCHES]++;}
#endif
    const int last=through*R.steps+R.reach;
    int i=S->interval,k=S->step;
    {
        // Per lab axis the first output whose window has not ended before its interval.
        int next[3];
        #pragma unroll
        for(int a=0;a<3;a++){const int before=i==0?-1:(i-1)*R.steps;int o=max(0,floordiv(before-R.half[a],R.steps));while(window_end(R,a,o)<=before)o++;next[a]=o;}
        for(bool spent=false;i<until&&!spent;i++){
            const int lo=i==0?-1:(i-1)*R.steps,hi=i==0?0:min(i*R.steps,last);
            // The interval's steps left, their windows' sums (while held) and rings.
            while(k<hi&&!spent){
                k++;
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
                spent=__shfl_sync(FULL,clock64()-launched>=R.budget?1:0,0)!=0;
            }
            if(k<hi)break;
            // The held twin's steps of the interval, up to the last output.
            if(held&&i>0&&i<=through){
                const long long begun=clock64();
                for(int m=lo+1;m<=hi;m++)hold_step(P,R,W,h,t.index,lane);
                twin_cycles+=clock64()-begun;
            }
            // The record of output i.
            double* out=records+(size_t)(i%MOST_INTERVALS)*SLOT_RECORD;
            for(int x=lane;x<SLOT_RECORD;x+=32)out[x]=0.;
            __syncwarp();
            SPAN_BEGIN(readout);
            if(i<=through&&t.death<0){
                record(P,W,t,lane,out);
                __syncwarp();
                for(int k=0;k<R.nlisted;k++)if(R.listed[k]==i){
                    double* sample=R.samples+((size_t)slot*R.nlisted+k)*7;double r[3],kappa[3];lab(P,t.q,r);lab(P,t.p,kappa);
                    if(lane<3){sample[lane]=r[lane];sample[3+lane]=kappa[lane];}if(lane==0)sample[6]=1.;
                }
            }
            if(held&&i<=through){const long long begun=clock64();__syncwarp();hold_record(P,R,W,h,lane,out);twin_cycles+=clock64()-begun;}
            SPAN_END(SPAN_READOUT,readout);SPAN_BEGIN(windows);
            // The windows ending in the interval: their means while held at their output.
            #pragma unroll
            for(int a=0;a<3;a++)for(int c=0;c<2;c++){
                const int o=next[a];
                if(o>through||window_end(R,a,o)>hi)break;
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
#ifdef PROFILE
    if(lane==0)W.profile[SPAN_TWIN]+=twin_cycles;
    SPAN_END(SPAN_TOTAL,launch);
    const long long storing=clock64();
#endif
    {
        if(lane==0){
            #pragma unroll
            for(int a=0;a<3;a++){S->q[a]=t.q[a];S->p[a]=t.p[a];}
            S->time=t.time;S->logn=t.logn;S->threshold=t.threshold;S->log_threshold=t.log_threshold;
            S->cluster=t.cluster;S->jumps=t.jumps;S->death=t.death;S->beats=t.beats;S->interval=i;S->step=k;
        }
        if(lane<15)S->prefix[lane]=prefix;
        if(lane<MAXM)S->psi[lane]=W.psi[lane];
        if(lane<MAXN)S->twin[lane]=W.twin[lane];
        if(lane==0){
            S->twin_logn=h.logn;S->twin_threshold=h.threshold;S->twin_log_threshold=h.log_threshold;S->twin_cluster=h.cluster;S->twin_jumps=h.jumps;
            R.costs[3*slot]=(unsigned long long)twin_cycles;R.costs[3*slot+1]=(unsigned long long)(clock64()-launched);R.costs[3*slot+2]=i>=intervals?PAST_ALL:(unsigned long long)i;
        }
    }
#ifdef PROFILE
    if(lane==0)W.profile[SPAN_CHECKPOINT]+=clock64()-storing;
    if(lane<SPANS)R.profile[(size_t)slot*SPANS+lane]+=(unsigned long long)W.profile[lane];
#endif
}

// The sums of the intervals [begin, end), which every slot has passed (`motion3d`'s records): per task of the wave one block, its
// records' sums per interval in the trajectories' order -- the CPU's order within a task.
extern "C" __global__ void task_sums(const Run R){
    const int through=R.tasks[3*blockIdx.x],count=R.tasks[3*blockIdx.x+1],first=R.tasks[3*blockIdx.x+2];
    const int last=through*R.steps+R.reach;
    int start[3];
    #pragma unroll
    for(int a=0;a<3;a++){const int before=R.begin==0?-1:(R.begin-1)*R.steps;int o=max(0,floordiv(before-R.half[a],R.steps));while(window_end(R,a,o)<=before)o++;start[a]=o;}
    // The block's sums of the intervals in the trajectories' order: each 128 threads every
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
            for(int a=0;a<3;a++){closes[a]=0;while(closes[a]<2&&next[a]<=through&&window_end(R,a,next[a])<=hi)outputs[a][closes[a]++]=next[a]++;}
            if(((i-R.begin)&1)!=group)continue;
            const double* rec=base+(size_t)(i%MOST_INTERVALS)*SLOT_RECORD;
            if(x<POINT_RECORD){
                if(i<=through){double s=0.;for(int w=0;w<count;w++)s+=rec[w*stride+x];R.sums[((size_t)blockIdx.x*R.outputs+i)*RECORD+x]=s;}
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
    point<true>(P,W,t,lane);exact_components(P,W,P.clusters[t.cluster],lane,X);
}
// A check's exact responses (tests/gpu.rs): a trajectory's cluster's photon tables at its point
// (`exact_responses`) into X, their rows' and the mixed blocks' shifts' room after them, one warp; NaN at X[0] where a resolvent is singular.
extern "C" __global__ void response_probe(const Program P,const State* S,float2* X){
    __shared__ Warp W;const int lane=threadIdx.x&31;if(threadIdx.x>=32)return;
    Traj t;
    #pragma unroll
    for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];}
    t.time=S->time;t.cluster=S->cluster;
    point<false>(P,W,t,lane);
    if(!exact_responses(P,W,P.clusters[t.cluster],lane,X,X+P.nnear*MAXN*3*NG,X+P.nnear*MAXN*3*NG+NE*32)&&lane==0)X[0]=make_float2(__int_as_float(0x7fc00000),0.f);
}
// A check's jump (tests/gpu.rs): a trajectory's state, a jump at its position and time (the first order
// in the Doppler shift, or the exact response where `exact`), the state it leaves.
extern "C" __global__ void jump_probe(const Program P,const Run R,State* S,float2* scratch,int exact){
    __shared__ Warp W;const int lane=threadIdx.x&31;if(threadIdx.x>=32)return;
    Traj t;
    #pragma unroll
    for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];}
    t.time=S->time;t.logn=S->logn;t.threshold=S->threshold;t.log_threshold=S->log_threshold;t.cluster=S->cluster;t.jumps=S->jumps;
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
    out[16]=MOST_INTERVALS;out[17]=SLOT_RECORD;out[18]=RECORD;out[19]=WARPS;out[20]=MAXN;out[21]=TWIN;out[22]=MOTION;out[23]=HARMONIC;out[24]=sizeof(Fourth);out[25]=sizeof(NearUnit);out[26]=sizeof(FirstField);out[27]=MOST_FIRST;
}
