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
// diagonal, the motion windows and every sum.
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
// A launch runs every task of a wave over a range of output intervals (src/gpu/mod.rs keeps it short
// of the display watchdog); the trajectories' states, motion windows and samples stay in device
// memory between launches. At every interval's end the block's warps meet: each one's record of the
// output and of the motion windows closing there is summed over the block in the trajectories'
// order -- the CPU's order within a task -- into the task's sums.

typedef unsigned int u32;
typedef unsigned long long u64;

#define MAXN 8                          // the most states of a cluster the kernel holds
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
#define RECORD 47                       // an output's sums: held, profile, doppler, 32 populations, 3 × (κ, κ², r, r²)
#define POINT_RECORD 35                 // the record's part taken at the output time
#define MOTION 35                       // where the motion windows' means start
#define FULL 0xffffffffu
#define TAU 6.283185307179586
#define PURPOSE_JUMP 3u                 // rng.rs `Purpose::Jump`

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

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(FULL,v,o);
    return v;
}
__device__ __forceinline__ float2 warp_sum(float2 v){return make_float2(warp_sum(v.x),warp_sum(v.y));}
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
    float profile;                      // its weight when taken at its peak (model.rs `profile_error`); −1 otherwise
    int carrier,near;                   // its carrier and its place among the near beams; −1 when not near
    u32 coupled;                        // the ground columns its coupling reaches (bit b)
    float coefficients[MOST_RANK][2];   // c_lk on its carrier's basis
};
struct Carrier{int rank,beams,first,pad;double detuning;};   // its beams: beam_list[first, first + beams)
// A cluster: its tables' place (node 0 at tables[table], `size` entries per node) and slices, its
// groups, classes and coherent groups, its carriers' photon tables (carrier_tables[carriers + 3c +
// (0 J, 1 J2, 2 T)]) and its states' ground positions.
struct Cluster{int n,size,table,h0,groups,ngroups,classes,nclasses,coherent,ncoherent,carriers,pad;int states[MAXN];};
// A kept group (model.rs `GroupLayout`): its field pairs pairs[pairs, pairs + npairs) and the S and Q slices.
struct Group{int absorbed,emitted,hermitian,pairs,npairs,s,q,pad;double residual;};
// A jump class (ground.rs `Class`): its photons photons[photons, photons + nphotons), the carriers they come from (bits).
struct Class{int target,photons,nphotons,carriers;double wave_number;};
// A photon: its carrier, its source column and ground row in the jump tables, its row in the target cluster, ω̃ − ω̃₀.
struct Photon{int carrier,col,to,row;double residual;};
struct Coherent{int channels,count;};
struct Channel{int carrier,col;double residual;};

struct Program{
    const Beam* beams;const int* beam_list;const Carrier* carriers;const Cluster* clusters;const Group* groups;const int2* pairs;
    const Class* classes;const Photon* photons;const Coherent* coherent;const Channel* channels;const int* carrier_tables;
    const float2* tables;
    // The exact response (model.rs `Exact`): K = H_ee − (i/2)D off its diagonal, its base and per
    // focused beam (NE × NE each); the diagonals in double precision, per block [NG ground | NE
    // excited] (ground ground energies E_b, excited Re K_ii); every beam's coupling V_l (NE × NG);
    // the emission C_a (3 × NG × NE).
    const float2* excited;const double* diagonal;const float2* couplings;const float2* emission;
    int nbeams,ncarriers,nclusters,focused,nodes,fields,nnear,doppler,loss,pad;
    int focused_beam[MOST_FOCUSED],stride[MOST_FOCUSED];
    int field_carrier[MOST_CARRIERS*MOST_RANK],field_rank[MOST_CARRIERS*MOST_RANK];
    int ground_states[NG],excited_states[NE];
    float hm,light2,allowance,truncation;   // ħ/m (um²/us), twice the light shift's bound, the first order's allowance, the exponentials' truncation
    float axes32[3][3],stiffness32[3];      // the axes and the stiffness below, in single precision
    double hm64,dt;
    double axes[3][3],omega[3],stiffness[3];// the trap's principal axes (rows), ω_i (rad/us), ω_i²/(ħ/m)
    double half_flow[3][3];                 // the flow for dt/2 per axis: cos ωτ, (ħ/m) sin ωτ/ω, (m/ħ) ω sin ωτ
};

// A trajectory's state between launches.
struct State{
    double q[3],p[3];                   // position (um) and momentum over ħ (1/um) on the trap's principal axes
    double time,logn,threshold,log_threshold;
    double prefix[15];                  // per lab axis: steps, Σκ, Σκ², Σr, Σr² through the last step (lib.rs `Windows`)
    float2 psi[MAXN];                   // ψ/‖ψ‖ on its cluster
    float doppler;int cluster,jumps,death;  // death: the step it was lost at, −1 while held
    u32 index;int pad;
};
// A launch: a wave's tasks over the output intervals [begin, end).
struct Run{
    State* states;double* rings;double* samples;double* sums;float2* scratch;const int* tasks;
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
};
// A jump's working space.
struct JumpSpace{float2 u[3][MAXN],f[MAXN];float weights[MOST_CLASSES],shares[MOST_BEAMS];int beams[MOST_BEAMS];};
struct Warp{
    Point pt;
    float2 psi[MAXN],start[MAXN],readout[MAXN];   // ψ, ψ at a step's start, the readout's Y ψ
    union{float2 m[3][MAXN*MAXN];float2 gradient_a[MOST_BEAMS][3];JumpSpace jump;};
    double record[POINT_RECORD];
    double close[3][2][5];              // per lab axis, per window closing in the interval: held, ⟨κ⟩, ⟨κ²⟩, ⟨r⟩, ⟨r²⟩
};

// The trajectory's scalars (uniform over the warp).
struct Traj{double q[3],p[3],time,logn,threshold,log_threshold;float doppler;int cluster,jumps,death;u32 index;};

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
// the focused beams' intensities, their gradients and stencils (and with the trap beam off), the
// carriers' fields.
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
        const int l=P.focused_beam[lane];const float2 a=pt.a[l];const float s=fminf(norm2(a),1.f);pt.s[lane]=s;
        #pragma unroll
        for(int i=0;i<3;i++){const float2 g=W.gradient_a[l][i];pt.grad_s[lane][i]=2.f*(a.x*g.x+a.y*g.y);}
        cubic(P.nodes,s,pt.first[lane],pt.weight[lane],pt.slope[lane]);
        float unused[4];cubic(P.nodes,lane==0?0.f:s,pt.far_first[lane],pt.far_weight[lane],unused);
    }
    if(lane<P.fields){
        const int c=P.field_carrier[lane],k=P.field_rank[lane];const Carrier C=P.carriers[c];
        float2 e=zero2(),d=zero2(),g[3]={zero2(),zero2(),zero2()};
        for(int j=0;j<C.beams;j++){
            const int l=P.beam_list[C.first+j];const float2 x=make_float2(P.beams[l].coefficients[k][0],P.beams[l].coefficients[k][1]);
            const float2 ax=pt.a[l]*x;e+=ax;d+=pt.shift[l]*ax;
            #pragma unroll
            for(int i=0;i<3;i++)g[i]+=W.gradient_a[l][i]*x;
        }
        Fields& f=pt.fields[c];f.e[k]=e;f.doppler[k]=d;
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

// ---------------------------------------------------------------------------------------------
// The generator.

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
        const Group G=P.groups[C.groups+gi];const Fields& fa=pt.fields[G.absorbed];const Fields& fe=pt.fields[G.emitted];
        const float2 phase=cis(G.residual*t);
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
    __syncwarp();
    #pragma unroll
    for(int j=0;j<SLOTS;j++)if(E.has(j,lane)){
        float2 h=make_float2(L.h0[j].x-m[j].x+0.5f*g[j].y,L.h0[j].y-m[j].y-0.5f*g[j].x);
        if(E.row[j]==E.col[j])h.x-=scalar;
        W.m[0][lane+32*j]=h;
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

// ψ ← U ψ (U in W.m[2]), renormalized; returns ‖Uψ‖² relative to the ‖ψ‖² = 1 it had.
__device__ float apply(Warp& W,int n,int lane){
    float2 v=zero2();
    if(lane<n){
        const float2* u=W.m[2]+lane*n;
        #pragma unroll
        for(int b=0;b<MAXN;b++)if(b<n)v+=u[b]*W.psi[b];
    }
    const float s=warp_sum(lane<n?norm2(v):0.f);
    __syncwarp();if(lane<n)W.psi[lane]=(1.f/sqrtf(s))*v;__syncwarp();
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

// The harmonic flow for τ along each principal axis (engine.rs `flow`): (x, κ) turned by ωτ in
// phase space; free flight when ω is 0.
__device__ void flow(const Program& P,Traj& t,double tau){
    const bool half=tau==0.5*P.dt;
    #pragma unroll
    for(int i=0;i<3;i++){
        const double w=P.omega[i];double c,sh,sw;
        if(half){c=P.half_flow[i][0];sh=P.half_flow[i][1];sw=P.half_flow[i][2];}
        else if(w==0.){c=1.;sh=P.hm64*tau;sw=0.;}
        else{double s;sincos(w*tau,&s,&c);sh=P.hm64/w*s;sw=w/P.hm64*s;}
        const double x=t.q[i],k=t.p[i];t.q[i]=x*c+k*sh;t.p[i]=k*c-x*sw;
    }
}

// One step's flow, kick and internal evolution of length dt without jumps (engine.rs `advance`):
// ln‖ψ‖² at the middle (`lh`) and the end (t.logn), the midpoint's point in W.pt and h0 there in L.
__device__ void advance(const Program& P,Warp& W,Traj& t,double dt,int lane,double& lh,Local& L){
    flow(P,t,0.5*dt);
    point(P,W,t,lane);
    const Cluster C=P.clusters[t.cluster];const Entries E(C.n,lane);
    local(P,W,C,E,t.time+0.5*dt,lane,L);
    exponential(P,W,E,0.5*dt,lane);
    const float nh=apply(W,C.n,lane);lh=t.logn+(double)logf(nh);
    // The Ehrenfest force −∇Re⟨ψ|H_eff|ψ⟩/⟨ψ|ψ⟩ at the middle, less the harmonic part the flow takes.
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
    for(int i=0;i<3;i++)t.p[i]+=(double)(tau*(P.axes32[i][0]*force[0]+P.axes32[i][1]*force[1]+P.axes32[i][2]*force[2]+P.stiffness32[i]*(float)t.q[i]));
    const float n1=apply(W,C.n,lane);t.logn=lh+(double)logf(n1);
    flow(P,t,0.5*dt);
    t.time+=dt;
}

// ---------------------------------------------------------------------------------------------
// The jump.

__device__ __forceinline__ int exact_at(int near,int col,int a,int to){return ((near*MAXN+col)*3+a)*NG+to;}

// x = A⁻¹ b for the NE × (NE + 1) augmented matrix [A | b] (row stride NE + 1) in place, by
// Gaussian elimination with partial pivoting (linear.rs `Lu`: the largest |a|², the last on ties);
// x into the last column. False when a pivot vanishes.
__device__ __noinline__ bool solve(float2* A,int lane){
    const int S=NE+1;
    for(int k=0;k<NE;k++){
        float best=-1.f;int p=k;if(lane>=k&&lane<NE){best=norm2(A[lane*S+k]);p=lane;}
        #pragma unroll
        for(int o=16;o>0;o>>=1){const float ob=__shfl_xor_sync(FULL,best,o);const int op=__shfl_xor_sync(FULL,p,o);if(ob>best||(ob==best&&op>p)){best=ob;p=op;}}
        if(!(best>0.f))return false;
        if(p!=k)for(int j=lane;j<S;j+=32){const float2 x=A[k*S+j];A[k*S+j]=A[p*S+j];A[p*S+j]=x;}
        __syncwarp();
        const float2 pivot=A[k*S+k];const int w=S-(k+1);
        for(int e=lane;e<(NE-1-k)*w;e+=32){const int i=k+1+e/w,j=k+1+e%w;A[i*S+j]-=cdiv(A[i*S+k],pivot)*A[k*S+j];}
        __syncwarp();
    }
    for(int i=NE-1;i>=0;i--){
        float2 s=zero2();for(int j=i+1+lane;j<NE;j+=32)s+=A[i*S+j]*A[j*S+NE];s=warp_sum(s);
        const float2 x=cdiv(A[i*S+NE]-s,A[i*S+i]);
        __syncwarp();if(lane==0)A[i*S+NE]=x;__syncwarp();
    }
    return true;
}

// The exact response's photon tables at the point (model.rs `Exact`, ground.rs `response`):
// X_l[a][to][col] = (C_a R_l(w_l))[to, b] for every near beam l and cluster column (b its ground
// state), R_l[:, b] = (K + w_l − E_b − δ_c)⁻¹ V_l[:, b] at the point's intensities. False when a
// resolvent is singular, as `Exact::new` fails (the jump keeps the first order).
__device__ __noinline__ bool exact_responses(const Program& P,const Warp& W,const Cluster& C,int lane,float2* X){
    float2* A=X+P.nnear*MAXN*3*NG;const int S=NE+1;const Point& pt=W.pt;
    for(int l=0;l<P.nbeams;l++){
        const Beam& b=P.beams[l];if(b.near<0)continue;
        for(int col=0;col<C.n;col++){
            const int g=C.states[col];float2* out=X+exact_at(b.near,col,0,0);
            if(!((b.coupled>>g)&1u)){if(lane<3*NG)out[lane]=zero2();__syncwarp();continue;}
            double shift=(double)pt.shift[l]-P.carriers[b.carrier].detuning-P.diagonal[g];
            for(int f=0;f<P.focused;f++)shift-=(double)pt.s[f]*P.diagonal[(1+f)*(NG+NE)+g];
            for(int e=lane;e<NE*NE;e+=32){
                const int i=e/NE,j=e-i*NE;float2 k=P.excited[e];
                for(int f=0;f<P.focused;f++)k+=pt.s[f]*P.excited[(1+f)*NE*NE+e];
                if(i==j){double d=P.diagonal[NG+i]+shift;for(int f=0;f<P.focused;f++)d+=(double)pt.s[f]*P.diagonal[(1+f)*(NG+NE)+NG+i];k.x=(float)d;}
                A[i*S+j]=k;
            }
            if(lane<NE)A[lane*S+NE]=P.couplings[(l*NE+lane)*NG+g];
            __syncwarp();
            if(!solve(A,lane))return false;
            if(lane<3*NG){const int a=lane/NG,to=lane-a*NG;float2 s=zero2();for(int i=0;i<NE;i++)s+=P.emission[(a*NG+to)*NE+i]*A[i*S+NE];out[lane]=s;}
            __syncwarp();
        }
    }
    return true;
}

// Every photon's matrix element of the cluster's jumps at the point (model.rs `Model::photons`),
// interpolated once per jump: M_a[to, from] = Σ_k E_k J_k − E^D_k J2_k, or the exact responses'
// Σ_l a_l C_a R_l(w_l), per photon and axis (the cluster's photons, all its classes', in order).
__device__ void photon_matrices(const Program& P,const Warp& W,const Cluster& C,int lane,const float2* X,float2* out){
    const float2* tables=P.tables+C.table;const int first=P.classes[C.classes].photons;
    const Class last=P.classes[C.classes+C.nclasses-1];const int count=3*(last.photons+last.nphotons-first);
    for(int j=lane;j<count;j+=32){
        const int a=j%3;const Photon ph=P.photons[first+j/3];const Carrier R=P.carriers[ph.carrier];float2 m=zero2();
        if(X==nullptr){
            const Fields& f=W.pt.fields[ph.carrier];const int j1=P.carrier_tables[C.carriers+3*ph.carrier],j2=P.carrier_tables[C.carriers+3*ph.carrier+1];
            #pragma unroll
            for(int k=0;k<MOST_RANK;k++)if(k<R.rank){const int at=((k*3+a)*NG+ph.to)*C.n+ph.col;m+=f.e[k]*entry(P,W,tables+j1+at,C.size,-1)-f.doppler[k]*entry(P,W,tables+j2+at,C.size,-1);}
        }else for(int b=0;b<R.beams;b++){const int l=P.beam_list[R.first+b];m+=X[exact_at(P.beams[l].near,ph.col,a,ph.to)]*W.pt.a[l];}
        out[j]=m;
    }
    __syncwarp();
}
// The amplitude u_a of a class at a row of its target cluster (model.rs `Model::photons`): over its
// photons landing in that row, e^{iω̃t} M_a[to, from] ψ[from], M from the photons' matrices.
__device__ float2 amplitude(const Program& P,const Warp& W,const Cluster& C,const Class& K,int a,int row,double t,const float2* M){
    const int first=P.classes[C.classes].photons;float2 u=zero2();
    for(int i=0;i<K.nphotons;i++){
        const Photon ph=P.photons[K.photons+i];if(ph.row!=row)continue;
        u+=cis(ph.residual*t)*M[3*(K.photons+i-first)+a]*W.psi[ph.col];
    }
    return u;
}

// ∇u_a of a class at a row (model.rs `Photons::gradient`): the fields' dependence on r and the
// intensities', from the jump tables J.
__device__ void amplitude_gradient(const Program& P,const Warp& W,const Cluster& C,const Class& K,int a,int row,double t,float2 du[3]){
    const float2* tables=P.tables+C.table;const Point& pt=W.pt;
    #pragma unroll
    for(int x=0;x<3;x++)du[x]=zero2();
    for(int i=0;i<K.nphotons;i++){
        const Photon ph=P.photons[K.photons+i];if(ph.row!=row)continue;
        const float2 phase=cis(ph.residual*t)*W.psi[ph.col];const Fields& f=pt.fields[ph.carrier];
        const int j1=P.carrier_tables[C.carriers+3*ph.carrier];const int rank=P.carriers[ph.carrier].rank;
        for(int k=0;k<rank;k++){
            const float2* x=tables+j1+((k*3+a)*NG+ph.to)*C.n+ph.col;const float2 j=entry(P,W,x,C.size,-1);
            #pragma unroll
            for(int d=0;d<3;d++)du[d]+=phase*f.grad[k][d]*j;
            for(int g=0;g<P.focused;g++){
                const float2 dj=entry(P,W,x,C.size,g);
                #pragma unroll
                for(int d=0;d<3;d++)du[d]+=pt.grad_s[g][d]*(phase*f.e[k]*dj);
            }
        }
    }
}

// Beam l's share of a class at a row (model.rs `Photons::shares`, along the polarization e): over
// the class's photons of carrier c landing in that row, e^{iω̃t} ψ[from] a_l Σ_a e_a z_a with
// z = Σ_k c_lk (J_k − w_l J2_k)[to, from], or the exact C_a R_l.
__device__ float2 share(const Program& P,const Warp& W,const Cluster& C,const Class& K,int c,int l,int row,double t,const float2* X,const double e[3]){
    const float2* tables=P.tables+C.table;const Beam& b=P.beams[l];const float2 al=W.pt.a[l];const float wl=W.pt.shift[l];
    const int rank=P.carriers[c].rank;const int j1=P.carrier_tables[C.carriers+3*c],j2=P.carrier_tables[C.carriers+3*c+1];
    float2 g=zero2();
    for(int i=0;i<K.nphotons;i++){
        const Photon ph=P.photons[K.photons+i];if(ph.row!=row||ph.carrier!=c)continue;
        const float2 phase=cis(ph.residual*t)*W.psi[ph.col]*al;float2 z=zero2();
        #pragma unroll
        for(int a=0;a<3;a++){
            float2 za=zero2();
            if(X==nullptr)for(int k=0;k<rank;k++){
                const int at=((k*3+a)*NG+ph.to)*C.n+ph.col;
                za+=make_float2(b.coefficients[k][0],b.coefficients[k][1])*(entry(P,W,tables+j1+at,C.size,-1)-wl*entry(P,W,tables+j2+at,C.size,-1));
            }else za=X[exact_at(b.near,ph.col,a,ph.to)];
            z+=(float)e[a]*za;
        }
        g+=phase*z;
    }
    return g;
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
__device__ __noinline__ void jump(const Program& P,const Run& R,Warp& W,Traj& t,bool exact,int lane,float2* scratch){
    t.jumps++;Rng rng(R.seed,PURPOSE_JUMP,t.index,(u32)t.jumps);
    point(P,W,t,lane);
    const Cluster C=P.clusters[t.cluster];
    {
        float s=0.f;for(int b=0;b<C.n;b++)s+=norm2(W.psi[b]);const float scale=1.f/sqrtf(s);
        __syncwarp();if(lane<C.n)W.psi[lane]=scale*W.psi[lane];__syncwarp();
    }
    const float2* X=exact&&exact_responses(P,W,C,lane,scratch)?scratch:nullptr;
    float2* M=scratch+P.nnear*MAXN*3*NG+NE*(NE+1);photon_matrices(P,W,C,lane,X,M);
    JumpSpace& J=W.jump;
    // The class, and the cluster it lands the atom in: by Σ_a ‖u_a‖².
    double total=0.;
    for(int c=0;c<C.nclasses;c++){
        const Class K=P.classes[C.classes+c];const int nt=P.clusters[K.target].n;
        float v=0.f;if(lane<3*nt)v=norm2(amplitude(P,W,C,K,lane%3,lane/3,t.time,M));
        const float w=warp_sum(v);if(lane==0)J.weights[c]=w;total+=(double)w;
    }
    __syncwarp();
    if(!(total>0.)){t.threshold=rng.open_uniform();t.log_threshold=log(t.threshold);return;}
    int pick=C.nclasses-1;
    {double x=rng.uniform()*total;for(int c=0;c<C.nclasses;c++){const double w=(double)J.weights[c];if(x<w){pick=c;break;}x-=w;}}
    const Class K=P.classes[C.classes+pick];const int nt=P.clusters[K.target].n;
    if(lane<3*nt)J.u[lane%3][lane/3]=amplitude(P,W,C,K,lane%3,lane/3,t.time,M);
    __syncwarp();
    // The photon's direction from the dipole pattern tr G − n·Re(G)·n of G_ab = ⟨u_a|u_b⟩.
    double gram[3][3];
    for(int a=0;a<3;a++)for(int b=0;b<3;b++){float s=0.f;for(int i=0;i<nt;i++){const float2 x=J.u[a][i],y=J.u[b][i];s+=x.x*y.x+x.y*y.y;}gram[a][b]=(double)s;}
    const double trace=gram[0][0]+gram[1][1]+gram[2][2];double n[3];
    for(;;){
        rng.direction(n);double q=0.;for(int a=0;a<3;a++)for(int b=0;b<3;b++)q+=n[a]*gram[a][b]*n[b];
        if(rng.uniform()*trace<=trace-q)break;
    }
    // Its polarization across it, from ‖Σ_a ε_a u_a‖² on a real orthonormal pair.
    double pair[2][3];across(n,pair);float weights[2];float2 f[2]={zero2(),zero2()};
    for(int j=0;j<2;j++){
        if(lane<nt)for(int a=0;a<3;a++)f[j]+=(float)pair[j][a]*J.u[a][lane];
        weights[j]=warp_sum(lane<nt?norm2(f[j]):0.f);
    }
    const int chosen=rng.uniform()*((double)weights[0]+(double)weights[1])<(double)weights[0]?0:1;
    const double* e=pair[chosen];const float size=weights[chosen];
    if(lane<nt)J.f[lane]=f[chosen];
    __syncwarp();
    // The recoils: the photon's, the absorbed beam's and the exact conditional mean's correction.
    double kick[3];for(int a=0;a<3;a++)kick[a]=-K.wave_number*n[a];
    double mean[3];
    {
        float m[3]={0.f,0.f,0.f};
        if(lane<3*nt){
            const int a=lane%3,row=lane/3;float2 du[3];amplitude_gradient(P,W,C,K,a,row,t.time,du);const float2 fr=J.f[row];
            #pragma unroll
            for(int x=0;x<3;x++){const float2 z=(float)e[a]*du[x];m[x]=fr.x*z.y-fr.y*z.x;}
        }
        #pragma unroll
        for(int x=0;x<3;x++)mean[x]=(double)warp_sum(m[x])/(double)size;
    }
    int count=0;double sum=0.;
    for(int c=0;c<P.ncarriers;c++){
        if(!((K.carriers>>c)&1))continue;const Carrier R=P.carriers[c];
        for(int j=0;j<R.beams;j++){
            const int l=P.beam_list[R.first+j];
            const float w=warp_sum(lane<nt?norm2(share(P,W,C,K,c,l,lane,t.time,X,e)):0.f);
            if(lane==0){J.shares[count]=w;J.beams[count]=l;}count++;sum+=(double)w;
        }
    }
    __syncwarp();
    if(sum>0.){
        double x=rng.uniform()*sum;int drawn=J.beams[count-1];
        for(int i=0;i<count;i++){const double w=(double)J.shares[i];if(x<w){drawn=J.beams[i];break;}x-=w;}
        double average[3]={0.,0.,0.};
        for(int i=0;i<count;i++){const Beam& b=P.beams[J.beams[i]];for(int a=0;a<3;a++)average[a]+=(double)J.shares[i]*((double)b.direction[a]*(double)b.k);}
        const Beam& b=P.beams[drawn];
        for(int a=0;a<3;a++)kick[a]+=(double)b.direction[a]*(double)b.k+mean[a]-average[a]/sum;
    }else for(int a=0;a<3;a++)kick[a]+=mean[a];
    #pragma unroll
    for(int i=0;i<3;i++)t.p[i]+=P.axes[i][0]*kick[0]+P.axes[i][1]*kick[1]+P.axes[i][2]*kick[2];
    const float scale=1.f/sqrtf(size);
    __syncwarp();if(lane<MAXN)W.psi[lane]=lane<nt?scale*J.f[lane]:zero2();__syncwarp();
    t.cluster=K.target;t.logn=0.;t.threshold=rng.open_uniform();t.log_threshold=log(t.threshold);
}

// ---------------------------------------------------------------------------------------------
// The step.

// The trap's potential of ψ at the point (model.rs `trap_potential`, with `bound` its upper bound
// `trap_bound`): Re⟨ψ|·|ψ⟩/⟨ψ|ψ⟩ of h0 at the point less with the trap beam off, and of −Φ (S less
// S with the trap beam off); the bound takes twice the light shift's bound for the latter.
__device__ float trap(const Program& P,const Warp& W,const Cluster& C,const Entries& E,const float2 (&h0)[SLOTS],double t,bool bound,int lane){
    const float2* tables=P.tables+C.table;const Point& pt=W.pt;
    float2 difference[SLOTS],far[SLOTS],slope[MOST_FOCUSED][SLOTS];
    gather<false>(P,W,true,tables+C.h0,C.size,E.nn,lane,far,slope);
    #pragma unroll
    for(int j=0;j<SLOTS;j++)difference[j]=h0[j]-far[j];
    if(bound)return quadratic(W,E,difference,lane)+P.light2;
    float2 here[SLOTS];
    for(int gi=0;gi<C.ngroups;gi++){
        const Group G=P.groups[C.groups+gi];const Fields& fa=pt.fields[G.absorbed];const Fields& fe=pt.fields[G.emitted];
        const float2 phase=cis(G.residual*t);
        for(int pi=0;pi<G.npairs;pi++){
            const int2 pair=P.pairs[G.pairs+pi];const int k1=pair.x,k=pair.y;
            const float2 ee=(G.hermitian&&k1<k?2.f:1.f)*(conj(fe.e[k1])*fa.e[k]);const float2 c=zero2()-ee*phase;
            gather<false>(P,W,false,tables+G.s+pi*E.nn,C.size,E.nn,lane,here,slope);
            gather<false>(P,W,true,tables+G.s+pi*E.nn,C.size,E.nn,lane,far,slope);
            #pragma unroll
            for(int j=0;j<SLOTS;j++)difference[j]+=c*(here[j]-far[j]);
        }
    }
    return quadratic(W,E,difference,lane);
}

// One step of dt (engine.rs `Engine::step`): flow, kick and flow; where ‖ψ‖² crosses the threshold
// the step is redone to the crossing, the jump made there and the rest taken from the state it left.
// True when the atom is lost at its end: its mechanical energy above the trap's far level.
__device__ bool step(const Program& P,const Run& R,Warp& W,Traj& t,int lane,float2* scratch){
    const float ratio=doppler_ratio(P,t,lane);const bool exact=ratio>P.allowance;
    if(!exact)t.doppler=fmaxf(t.doppler,ratio);
    double left=P.dt;Local L;bool fresh=false;
    for(;;){
        const Traj start=t;
        if(lane<MAXN)W.start[lane]=W.psi[lane];
        double lh;advance(P,W,t,left,lane,lh,L);
        if(t.logn>t.log_threshold)break;
        // The crossing on ln‖ψ‖², linear on the half where it lies; the step redone to it.
        double ta,tb,la,lb;
        if(lh<=t.log_threshold){ta=0.;tb=0.5*left;la=start.logn;lb=lh;}else{ta=0.5*left;tb=left;la=lh;lb=t.logn;}
        const double at=lb<la?fmin(fmax(ta+(t.log_threshold-la)/(lb-la)*(tb-ta),ta),tb):tb;
        t=start;__syncwarp();if(lane<MAXN)W.psi[lane]=W.start[lane];__syncwarp();
        if(at>0.){double unused;advance(P,W,t,at,lane,unused,L);}
        jump(P,R,W,t,exact,lane,scratch);
        left-=at;
        if(!(left>0.)){point(P,W,t,lane);fresh=true;break;}
    }
    if(!P.loss)return false;
    const double kinetic=0.5*P.hm64*(t.p[0]*t.p[0]+t.p[1]*t.p[1]+t.p[2]*t.p[2]);
    const Cluster C=P.clusters[t.cluster];const Entries E(C.n,lane);
    if(fresh){float2 slope[MOST_FOCUSED][SLOTS];gather<false>(P,W,false,P.tables+C.table+C.h0,C.size,E.nn,lane,L.h0,slope);}
    return kinetic+(double)trap(P,W,C,E,L.h0,t.time,true,lane)>0.&&kinetic+(double)trap(P,W,C,E,L.h0,t.time,false,lane)>0.;
}

// ---------------------------------------------------------------------------------------------
// The record of an output.

// R_c[i][col] = Σ_k E_k T_k[i][col] at the stencil (model.rs `populations`' responses).
__device__ float2 response(const Program& P,const Warp& W,const Cluster& C,int c,int i,int col){
    const float2* x=P.tables+C.table+P.carrier_tables[C.carriers+3*c+2]+i*C.n+col;const Fields& f=W.pt.fields[c];float2 r=zero2();
    for(int k=0;k<P.carriers[c].rank;k++)r+=f.e[k]*entry(P,W,x+k*NE*C.n,C.size,-1);
    return r;
}

// The held atom's record of an output (stats.rs `Group::add`): held, the profile error, the
// (kv/σ)², the 32 populations (model.rs `populations`: the bare ground ones ρ − ½{Y, ρ}, the
// excited ones Σ |Σ_coherent e^{−iε̃t} (R ψ)_i|²) at its point, into W.record.
__device__ void record(const Program& P,Warp& W,const Traj& t,int lane){
    point(P,W,t,lane);
    const Cluster C=P.clusters[t.cluster];const int n=C.n;double* out=W.record;
    float profile=0.f,ratio=0.f;
    if(lane<P.nbeams){
        const Beam& b=P.beams[lane];
        if(b.profile>=0.f)profile=fmaxf(1.f-norm2(W.pt.a[lane]),0.f)*b.profile;
        if(b.inv_pole>0.f){const float x=W.pt.shift[lane]*b.inv_pole;ratio=x*x;}
    }
    profile=warp_max(profile);ratio=warp_max(ratio);
    float norm=0.f;for(int b=0;b<n;b++)norm+=norm2(W.psi[b]);
    float2* y=W.readout;
    if(lane<MAXN)y[lane]=zero2();
    __syncwarp();
    // Each channel's response in its excited row, once: the first CACHED of a group in the matrices' space.
    constexpr int CACHED=3*MAXN*MAXN/NE;float2* cache=W.m[0];
    float excited=0.f;
    for(int gi=0;gi<C.ncoherent;gi++){
        const Coherent G=P.coherent[C.coherent+gi];float2 e=zero2();
        if(lane<NE)for(int h=0;h<G.count;h++){
            const Channel ch=P.channels[G.channels+h];const float2 r=response(P,W,C,ch.carrier,lane,ch.col);if(h<CACHED)cache[h*NE+lane]=r;
            e+=cis(-ch.residual*t.time)*r*W.psi[ch.col];
        }
        for(int h=0;h<G.count;h++){
            const Channel ch=P.channels[G.channels+h];float2 z=zero2();
            if(lane<NE)z=cis(ch.residual*t.time)*conj(h<CACHED?cache[h*NE+lane]:response(P,W,C,ch.carrier,lane,ch.col))*e;
            z=warp_sum(z);if(lane==0)y[ch.col]+=z;
            __syncwarp();
        }
        excited+=norm2(e)/norm;
    }
    __syncwarp();
    if(lane<3)out[lane]=lane==0?1.:lane==1?(double)profile:(double)ratio;
    if(lane<NE)out[3+P.excited_states[lane]]=(double)excited;
    if(lane<NG){
        const int x=lane<n?lane:-1;double v=0.;
        if(x>=0){const float2 p=W.psi[x],q=y[x];v=(double)((norm2(p)-(p.x*q.x+p.y*q.y))/norm);}
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
// takes the steps ((i − 1)·steps, i·steps], the last ones past the last output its reach.
extern "C" __global__ void __launch_bounds__(WARPS*32,2) motion3d(const Program P,const Run R){
    __shared__ Warp warps[WARPS];
    const int lane=threadIdx.x&31,wi=threadIdx.x>>5;Warp& W=warps[wi];
    const int count=R.tasks[3*blockIdx.x+1],slot=R.tasks[3*blockIdx.x+2]+wi;const bool active=wi<count;
    State* S=R.states+slot;double* ring=R.rings+(size_t)slot*R.ring_stride;float2* scratch=R.scratch+(size_t)slot*R.scratch_stride;
    Traj t;double prefix=0.;
    if(active){
        #pragma unroll
        for(int i=0;i<3;i++){t.q[i]=S->q[i];t.p[i]=S->p[i];}
        t.time=S->time;t.logn=S->logn;t.threshold=S->threshold;t.log_threshold=S->log_threshold;
        t.doppler=S->doppler;t.cluster=S->cluster;t.jumps=S->jumps;t.death=S->death;t.index=S->index;
        if(lane<15)prefix=S->prefix[lane];
        if(lane<MAXN)W.psi[lane]=S->psi[lane];
    }
    __syncwarp();
    const int last=(R.outputs-1)*R.steps+R.reach;
    // Per lab axis the first output whose window has not ended before this launch.
    int next[3];
    #pragma unroll
    for(int a=0;a<3;a++){const int before=R.begin==0?-1:(R.begin-1)*R.steps;int o=max(0,floordiv(before-R.half[a],R.steps));while(window_end(R,a,o)<=before)o++;next[a]=o;}
    for(int i=R.begin;i<R.end;i++){
        const int lo=i==0?-1:(i-1)*R.steps,hi=i==0?0:min(i*R.steps,last);
        // The interval's steps, their windows' sums (while held) and rings.
        if(active)for(int k=lo+1;k<=hi;k++){
            if(t.death<0){
                if(k>0&&step(P,R,W,t,lane,scratch))t.death=k;
                if(t.death<0&&lane<15){
                    const int a=lane/5,v=lane-5*a;float r[3],kappa[3];lab32(P,t.q,r);lab32(P,t.p,kappa);
                    const double x=(double)(v<3?kappa[a]:r[a]);prefix+=v==0?1.:v==1||v==3?x:x*x;
                }
            }
            if(lane<15)ring[ring_at(R,lane/5,k)+lane%5]=prefix;
        }
        // The record of output i.
        if(lane<POINT_RECORD)W.record[lane]=0.;
        __syncwarp();
        if(active&&i<R.outputs&&t.death<0){
            record(P,W,t,lane);
            for(int k=0;k<R.nlisted;k++)if(R.listed[k]==i){
                double* out=R.samples+((size_t)slot*R.nlisted+k)*7;double r[3],kappa[3];lab(P,t.q,r);lab(P,t.p,kappa);
                if(lane<3){out[lane]=r[lane];out[3+lane]=kappa[lane];}if(lane==0)out[6]=1.;
            }
        }
        // The windows ending in the interval: their means while held at their output.
        int outputs[3][2],closes[3];
        #pragma unroll
        for(int a=0;a<3;a++){closes[a]=0;while(closes[a]<2&&next[a]<R.outputs&&window_end(R,a,next[a])<=hi)outputs[a][closes[a]++]=next[a]++;}
        #pragma unroll
        for(int a=0;a<3;a++)for(int c=0;c<2;c++){
            double v=0.;
            if(active&&c<closes[a]&&lane<15&&lane/5==a){
                const int o=outputs[a][c],at=o*R.steps,h=min(R.half[a],at),end=at+h,start=at-h-1;
                const bool held=t.death<0||t.death>at;
                if(held){
                    const int x=lane-5*a;
                    const double now=end==hi?prefix:ring[ring_at(R,a,end)+x];
                    const double before=start<0?0.:ring[ring_at(R,a,start)+x];
                    v=now-before;
                }
            }
            // v: lane 5a + x holds the window's sum of x; the means divide by its steps (x = 0).
            const double steps=__shfl_sync(FULL,v,5*a);
            if(lane<15&&lane/5==a){const int x=lane-5*a;W.close[a][c][x]=x==0?(steps>0.?1.:0.):(steps>0.?v/steps:0.);}
        }
        __syncthreads();
        // The block's sums in the trajectories' order.
        const int block=blockIdx.x;
        if(i<R.outputs&&threadIdx.x<POINT_RECORD){
            double s=0.;for(int w=0;w<WARPS;w++)s+=warps[w].record[threadIdx.x];
            R.sums[((size_t)block*R.outputs+i)*RECORD+threadIdx.x]=s;
        }
        if(threadIdx.x>=64&&threadIdx.x<64+24){
            const int j=threadIdx.x-64,a=j/8,c=(j/4)%2,x=j%4;
            if(c<closes[a]){
                double s=0.;for(int w=0;w<WARPS;w++)if(warps[w].close[a][c][0]>0.)s+=warps[w].close[a][c][1+x];
                R.sums[((size_t)block*R.outputs+outputs[a][c])*RECORD+MOTION+4*a+x]=s;
            }
        }
        __syncthreads();
    }
    if(active){
        if(lane==0){
            #pragma unroll
            for(int i=0;i<3;i++){S->q[i]=t.q[i];S->p[i]=t.p[i];}
            S->time=t.time;S->logn=t.logn;S->threshold=t.threshold;S->log_threshold=t.log_threshold;
            S->doppler=t.doppler;S->cluster=t.cluster;S->jumps=t.jumps;S->death=t.death;
        }
        if(lane<15)S->prefix[lane]=prefix;
        if(lane<MAXN)S->psi[lane]=W.psi[lane];
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
}
