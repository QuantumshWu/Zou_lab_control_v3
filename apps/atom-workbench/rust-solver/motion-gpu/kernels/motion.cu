// The 2D motion engine's kernels (src/engine.rs drives them). Every kernel is a short pass over the
// trajectories' wavefunctions; nothing loops for long, so no kernel nears the display watchdog.
//
// Layout: trajectory t's wavefunction is `state[t][c][y][x]` (complex float2), c < 5 its internal
// components in its hyperfine block (F = 2: 5, F = 1: 3; a lost trajectory has none). A plane is
// one component's NY x NX grid.

typedef unsigned int u32;
typedef unsigned long long u64;

#define COMPONENTS 5
#define PI 3.141592653589793

__device__ __forceinline__ float2 cmul(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ __forceinline__ float2 cmulc(float2 a,float2 b){return make_float2(a.x*b.x+a.y*b.y,a.y*b.x-a.x*b.y);} // a * conj(b)
__device__ __forceinline__ float2 cadd(float2 a,float2 b){return make_float2(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ float2 csub(float2 a,float2 b){return make_float2(a.x-b.x,a.y-b.y);}
__device__ __forceinline__ float2 cscale(float2 a,float s){return make_float2(a.x*s,a.y*s);}
__device__ __forceinline__ float2 cconj(float2 a){return make_float2(a.x,-a.y);}
__device__ __forceinline__ float norm2(float2 a){return a.x*a.x+a.y*a.y;}
__device__ __forceinline__ float2 shfl_xor(float2 v,int mask){return make_float2(__shfl_xor_sync(0xffffffffu,v.x,mask),__shfl_xor_sync(0xffffffffu,v.y,mask));}

constexpr int log2i(int n){return n<=1?0:1+log2i(n/2);}

// ---------------------------------------------------------------------------------------------
// A length-N FFT (N a power of two) by a team of T = min(N, 32) lanes of one warp, each holding
// E = N/T elements: element j = e*T + t. `forward` is the DIF transform, natural order in,
// bit-reversed order out (position j holds X[bitrev(j)]), X[k] = sum_j x_j e^{-2 pi i jk/N};
// `inverse` is the DIT transform with conjugate twiddles, bit-reversed in, natural out, and
// inverse(forward(x)) = N x. The transform between them multiplies position j by a function of
// the frequency bitrev(j) (the tables are written in that order). Every lane of the warp must
// call both (they shuffle); for N < 32 a warp holds 32/N teams, and xor offsets < T stay inside a
// team. The twiddles are computed once per kernel into registers.
template<int N> struct Fft{
    static constexpr int T=N<32?N:32;
    static constexpr int E=N/T;
    static constexpr int LT=log2i(T);
    float2 local[E>1?E-1:1];   // the local stages' twiddles: stage of half-size h = hl*T at local[hl-1 + e%hl]
    float2 shuffle[LT>0?LT:1]; // the shuffle stages' twiddles: stage h at shuffle[log2 h]
    __device__ void init(int t){
        #pragma unroll
        for(int hl=1;hl<E;hl*=2){
            #pragma unroll
            for(int r=0;r<hl;r++){float s,c;sincospif(-(float)(r*T+t)/(float)(hl*T),&s,&c);local[hl-1+r]=make_float2(c,s);}
        }
        #pragma unroll
        for(int l=0;l<LT;l++){int h=1<<l;float s,c;sincospif(-(float)(t&(h-1))/(float)h,&s,&c);shuffle[l]=make_float2(c,s);}
    }
    __device__ void forward(float2 (&v)[E],int t) const{
        #pragma unroll
        for(int hl=E/2;hl>=1;hl/=2){
            #pragma unroll
            for(int e=0;e<E;e++)if((e%(2*hl))<hl){
                float2 a=v[e],b=v[e+hl];v[e]=cadd(a,b);v[e+hl]=cmul(csub(a,b),local[hl-1+e%hl]);
            }
        }
        #pragma unroll
        for(int l=LT-1;l>=0;l--){
            const int h=1<<l;const bool top=(t&h)==0;const float2 w=shuffle[l];
            #pragma unroll
            for(int e=0;e<E;e++){float2 p=shfl_xor(v[e],h);v[e]=top?cadd(v[e],p):cmul(csub(p,v[e]),w);}
        }
    }
    __device__ void inverse(float2 (&v)[E],int t) const{
        #pragma unroll
        for(int l=0;l<LT;l++){
            const int h=1<<l;const bool top=(t&h)==0;const float2 w=shuffle[l];
            #pragma unroll
            for(int e=0;e<E;e++){float2 p=shfl_xor(v[e],h);v[e]=top?cadd(v[e],cmulc(p,w)):csub(p,cmulc(v[e],w));}
        }
        #pragma unroll
        for(int hl=1;hl<E;hl*=2){
            #pragma unroll
            for(int e=0;e<E;e++)if((e%(2*hl))<hl){
                float2 a=v[e],b=cmulc(v[e+hl],local[hl-1+e%hl]);v[e]=cadd(a,b);v[e+hl]=csub(a,b);
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Per-trajectory records of an output time (floats, summed over the trajectory's points):
#define REC_GROUND 0     // 8 bare ground populations (rho - {Y, rho}/2), by block-ordered state (see Jumps)
#define REC_EXCITED 8    // 24 excited populations tr(E_i rho), by excited state
#define REC_NORM 32
#define REC_X 33         // <x>, <y>, <x^2>, <y^2> (um)
#define REC_KX 37        // <kx>, <ky>, <kx^2>, <ky^2> (rad/um)
#define REC_TAIL 41      // the parts of <x^2>, <y^2>, <kx^2>, <ky^2> beyond 0.7 of the grid's reach
#define RECORD 48

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int o=16;o>0;o/=2)v+=__shfl_xor_sync(0xffffffffu,v,o);
    return v;
}
// Sums v over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the sum.
__device__ float block_sum(float v,float* scratch){
    v=warp_sum(v);
    __syncthreads();
    if((threadIdx.x&31)==0)scratch[threadIdx.x>>5]=v;
    __syncthreads();
    float total=0.f;for(int w=0;w<(int)(blockDim.x>>5);w++)total+=scratch[w];
    return total;
}

// ---------------------------------------------------------------------------------------------
// The passes over a plane in shared memory, one block per plane:
// STEP     plane <- K_y K_x (scale * plane), K given per frequency position by `tx`, `ty` (with 1/N);
// OBSERVE  the same, adding the plane's momentum moments to the trajectory's record and its
//          weighted |psi(k)|^2 to the marginals (by frequency position);
// DERIVE   dx <- kappa_x plane, dy <- kappa_y plane (tx, ty: kappa/N), the plane unchanged.
#define MODE_STEP 0
#define MODE_OBSERVE 1
#define MODE_DERIVE 2
struct PlaneArgs{
    const float2* tx;const float2* ty;
    float2* dx;float2* dy;
    const float* kappa_x;const float* kappa_y;float tail_x,tail_y;
    float* record;float weight;float* marginal_x;float* marginal_y;
};
template<int NX,int NY,int MODE>
__device__ void plane_pass(float2* __restrict__ plane,float scale,const PlaneArgs& a,float2* sm,float* smarg){
    using FX=Fft<NX>;using FY=Fft<NY>;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5;
    constexpr int S=NX+1; // padded row stride: column reads are free of bank conflicts
    float m1=0.f,m2=0.f,mt=0.f;
    if(MODE==MODE_OBSERVE){for(int i=threadIdx.x;i<NX+NY;i+=blockDim.x)smarg[i]=0.f;__syncthreads();}
    {
        constexpr int teams=32/FX::T;const int t=lane%FX::T,team=lane/FX::T;
        FX f;f.init(t);
        float2 factor[FX::E];float kappa[FX::E];
        #pragma unroll
        for(int e=0;e<FX::E;e++){factor[e]=__ldg(a.tx+e*FX::T+t);kappa[e]=MODE==MODE_OBSERVE?__ldg(a.kappa_x+e*FX::T+t):0.f;}
        for(int r0=warp*teams;r0<NY;r0+=warps*teams){
            const int r=r0+team;const bool ok=r<NY;float2 v[FX::E];
            #pragma unroll
            for(int e=0;e<FX::E;e++)v[e]=ok?cscale(plane[r*NX+e*FX::T+t],scale):make_float2(0.f,0.f);
            if(MODE==MODE_DERIVE&&ok){
                #pragma unroll
                for(int e=0;e<FX::E;e++)sm[r*S+e*FX::T+t]=v[e];
            }
            f.forward(v,t);
            if(MODE==MODE_OBSERVE&&ok){
                #pragma unroll
                for(int e=0;e<FX::E;e++){const float p=norm2(v[e]),k=kappa[e];m1+=p*k;m2+=p*k*k;if(fabsf(k)>a.tail_x)mt+=p*k*k;atomicAdd(smarg+e*FX::T+t,p);}
            }
            #pragma unroll
            for(int e=0;e<FX::E;e++)v[e]=cmul(v[e],factor[e]);
            f.inverse(v,t);
            if(ok){
                #pragma unroll
                for(int e=0;e<FX::E;e++){if(MODE==MODE_DERIVE)a.dx[r*NX+e*FX::T+t]=v[e];else sm[r*S+e*FX::T+t]=v[e];}
            }
        }
    }
    if(MODE==MODE_OBSERVE){
        const float n=1.f/NX;m1=block_sum(m1,smarg+NX+NY)*n;m2=block_sum(m2,smarg+NX+NY)*n;mt=block_sum(mt,smarg+NX+NY)*n;
        if(threadIdx.x==0){atomicAdd(a.record+REC_KX,m1);atomicAdd(a.record+REC_KX+2,m2);atomicAdd(a.record+REC_TAIL+2,mt);}
        for(int i=threadIdx.x;i<NX;i+=blockDim.x)atomicAdd(a.marginal_x+i,a.weight*n*smarg[i]);
        m1=m2=mt=0.f;
    }
    __syncthreads();
    if constexpr(NY>1){
        constexpr int teams=32/FY::T;const int t=lane%FY::T,team=lane/FY::T;
        FY f;f.init(t);
        float2 factor[FY::E];float kappa[FY::E];
        #pragma unroll
        for(int e=0;e<FY::E;e++){factor[e]=__ldg(a.ty+e*FY::T+t);kappa[e]=MODE==MODE_OBSERVE?__ldg(a.kappa_y+e*FY::T+t):0.f;}
        for(int c0=warp*teams;c0<NX;c0+=warps*teams){
            const int c=c0+team;const bool ok=c<NX;float2 v[FY::E];
            #pragma unroll
            for(int e=0;e<FY::E;e++)v[e]=ok?sm[(e*FY::T+t)*S+c]:make_float2(0.f,0.f);
            f.forward(v,t);
            if(MODE==MODE_OBSERVE&&ok){
                #pragma unroll
                for(int e=0;e<FY::E;e++){const float p=norm2(v[e]),k=kappa[e];m1+=p*k;m2+=p*k*k;if(fabsf(k)>a.tail_y)mt+=p*k*k;atomicAdd(smarg+NX+e*FY::T+t,p);}
            }
            #pragma unroll
            for(int e=0;e<FY::E;e++)v[e]=cmul(v[e],factor[e]);
            f.inverse(v,t);
            if(ok){
                #pragma unroll
                for(int e=0;e<FY::E;e++)sm[(e*FY::T+t)*S+c]=v[e];
            }
        }
        if(MODE==MODE_OBSERVE){
            const float n=1.f/NY;m1=block_sum(m1,smarg+NX+NY)*n;m2=block_sum(m2,smarg+NX+NY)*n;mt=block_sum(mt,smarg+NX+NY)*n;
            if(threadIdx.x==0){atomicAdd(a.record+REC_KX+1,m1);atomicAdd(a.record+REC_KX+3,m2);atomicAdd(a.record+REC_TAIL+3,mt);}
            for(int i=threadIdx.x;i<NY;i+=blockDim.x)atomicAdd(a.marginal_y+i,a.weight*n*smarg[NX+i]);
        }
        __syncthreads();
    }
    float2* out=MODE==MODE_DERIVE?(NY>1?a.dy:nullptr):plane;
    if(out)for(int i=threadIdx.x;i<NX*NY;i+=blockDim.x)out[i]=sm[(i/NX)*S+i%NX];
}
// Shared memory: the plane, then (OBSERVE) NX + NY marginal bins and 32 reduction slots.
#define PLANE_KERNELS(NX,NY) \
extern "C" __global__ void __launch_bounds__(256) kinetic_##NX##x##NY(float2* state,const int* __restrict__ components,const float* __restrict__ scale,const float2* tx,const float2* ty){ \
    extern __shared__ float2 sm[]; \
    const int plane=blockIdx.x,t=plane/COMPONENTS,c=plane%COMPONENTS; \
    if(c>=components[t])return; \
    PlaneArgs a={};a.tx=tx;a.ty=ty; \
    plane_pass<NX,NY,MODE_STEP>(state+(u64)plane*(NX*NY),scale[t],a,sm,nullptr); \
} \
extern "C" __global__ void __launch_bounds__(256) observe_kinetic_##NX##x##NY(float2* state,const int* __restrict__ components,const float* __restrict__ scale,const float2* tx,const float2* ty, \
        const float* kappa_x,const float* kappa_y,float tail_x,float tail_y,float* records,const float* weights,float* marginal_x,float* marginal_y){ \
    extern __shared__ float2 sm[]; \
    const int plane=blockIdx.x,t=plane/COMPONENTS,c=plane%COMPONENTS; \
    if(c>=components[t])return; \
    PlaneArgs a={};a.tx=tx;a.ty=ty;a.kappa_x=kappa_x;a.kappa_y=kappa_y;a.tail_x=tail_x;a.tail_y=tail_y; \
    a.record=records+(u64)t*RECORD;a.weight=weights[t];a.marginal_x=marginal_x;a.marginal_y=marginal_y; \
    plane_pass<NX,NY,MODE_OBSERVE>(state+(u64)plane*(NX*NY),scale[t],a,sm,(float*)(sm+NY*(NX+1))); \
} \
/* kappa psi of the listed trajectories' planes (all when `list` is null; `count` of them, read */ \
/* on the device) into dx and dy at the trajectory's planes. */ \
extern "C" __global__ void __launch_bounds__(256) derive_##NX##x##NY(const float2* state,const int* __restrict__ components,const int* list,const int* count, \
        int trajectories,float2* dx,float2* dy,const float2* tx,const float2* ty){ \
    extern __shared__ float2 sm[]; \
    const int n=list?*count:trajectories; \
    for(int slot=blockIdx.x;slot<n*COMPONENTS;slot+=gridDim.x){ \
        const int t=list?list[slot/COMPONENTS]:slot/COMPONENTS,c=slot%COMPONENTS; \
        if(c>=components[t])continue; \
        const u64 p=((u64)t*COMPONENTS+c)*(NX*NY); \
        PlaneArgs a={};a.tx=tx;a.ty=ty;a.dx=dx+p;a.dy=dy+p; \
        plane_pass<NX,NY,MODE_DERIVE>((float2*)state+p,1.f,a,sm,nullptr); \
        __syncthreads(); \
    } \
}
PLANE_KERNELS(16,16)
PLANE_KERNELS(32,32)
PLANE_KERNELS(64,64)
PLANE_KERNELS(64,1)
PLANE_KERNELS(128,1)
PLANE_KERNELS(256,1)
PLANE_KERNELS(512,1)
PLANE_KERNELS(1024,1)

// ---------------------------------------------------------------------------------------------
// Local matrices at the grid points: block F = 2 (5 x 5) and F = 1 (3 x 3), entry (i, j) of
// point x at m[(i*n + j)*points + x].
struct Local{const float2* m5;const float2* m3;};

// psi <- M psi at one point of a trajectory's n components (stride `points`); returns |M psi|^2.
__device__ __forceinline__ float apply_local(float2* psi,u64 points,int n,const Local& m,u64 x){
    float n2=0.f;
    if(n==5){
        float2 a[5],b[5];
        #pragma unroll
        for(int i=0;i<5;i++)a[i]=psi[i*points];
        #pragma unroll
        for(int i=0;i<5;i++){float2 z=make_float2(0.f,0.f);
            #pragma unroll
            for(int j=0;j<5;j++)z=cadd(z,cmul(__ldg(m.m5+(u64)(i*5+j)*points+x),a[j]));b[i]=z;n2+=norm2(z);}
        #pragma unroll
        for(int i=0;i<5;i++)psi[i*points]=b[i];
    }else if(n==3){
        float2 a[3],b[3];
        #pragma unroll
        for(int i=0;i<3;i++)a[i]=psi[i*points];
        #pragma unroll
        for(int i=0;i<3;i++){float2 z=make_float2(0.f,0.f);
            #pragma unroll
            for(int j=0;j<3;j++)z=cadd(z,cmul(__ldg(m.m3+(u64)(i*3+j)*points+x),a[j]));b[i]=z;n2+=norm2(z);}
        #pragma unroll
        for(int i=0;i<3;i++)psi[i*points]=b[i];
    }
    return n2;
}

// Each block's per-trajectory sums (one per warp in `sums`) to partial[t*tiles + tile].
#define SUB 8
__device__ void store_partials(float (*sums)[8],int t0,int count,float* partial){
    __syncthreads();
    if((int)threadIdx.x<SUB&&(int)threadIdx.x<count){
        float total=0.f;for(int w=0;w<(int)(blockDim.x>>5);w++)total+=sums[threadIdx.x][w];
        partial[(u64)(t0+threadIdx.x)*gridDim.x+blockIdx.x]=total;
    }
}

// The local step psi <- V_F psi for every trajectory; grid (tiles, trajectories/SUB): each block
// takes 256 points of SUB trajectories, so its table entries are read once per SUB of them.
extern "C" __global__ void __launch_bounds__(256) local_step(float2* state,int points,int trajectories,const int* components,Local m,float* partial){
    __shared__ float sums[SUB][8];
    const int x=blockIdx.x*blockDim.x+threadIdx.x,t0=blockIdx.y*SUB;
    #pragma unroll 1
    for(int s=0;s<SUB;s++){
        const int t=t0+s;float n2=0.f;
        if(t<trajectories&&x<points)n2=apply_local(state+(u64)t*COMPONENTS*points+x,points,components[t],m,x);
        n2=warp_sum(n2);if((threadIdx.x&31)==0)sums[s][threadIdx.x>>5]=n2;
    }
    store_partials(sums,t0,min(SUB,trajectories-t0),partial);
}

// The Doppler term as an explicit step psi <- psi - i tau (W_x kappa_x psi + W_y kappa_y psi) with
// kappa psi from `derive`; the norm after it to partial, as `local_step`.
extern "C" __global__ void __launch_bounds__(256) doppler_kick(float2* state,const float2* dxs,const float2* dys,int points,int trajectories,const int* components,
        Local wx,Local wy,float tau,float* partial){
    __shared__ float sums[SUB][8];
    const int x=blockIdx.x*blockDim.x+threadIdx.x,t0=blockIdx.y*SUB;
    #pragma unroll 1
    for(int s=0;s<SUB;s++){
        const int t=t0+s;float n2=0.f;
        if(t<trajectories&&x<points){
            const u64 base=(u64)t*COMPONENTS*points+x;const int n=components[t];
            float2* psi=state+base;const float2* dx=dxs+base;const float2* dy=dys+base;
            const float2* mx=n==5?wx.m5:wx.m3;const float2* my=n==5?wy.m5:wy.m3;
            float2 a[5],b[5],c[5];
            for(int i=0;i<n;i++){a[i]=dx[(u64)i*points];b[i]=dy[(u64)i*points];}
            for(int i=0;i<n;i++){float2 z=make_float2(0.f,0.f);
                for(int j=0;j<n;j++){z=cadd(z,cmul(__ldg(mx+(u64)(i*n+j)*points+x),a[j]));z=cadd(z,cmul(__ldg(my+(u64)(i*n+j)*points+x),b[j]));}
                c[i]=z;}
            for(int i=0;i<n;i++){float2 p=psi[(u64)i*points];p=make_float2(p.x+tau*c[i].y,p.y-tau*c[i].x);psi[(u64)i*points]=p;n2+=norm2(p);}
        }
        n2=warp_sum(n2);if((threadIdx.x&31)==0)sums[s][threadIdx.x>>5]=n2;
    }
    store_partials(sums,t0,min(SUB,trajectories-t0),partial);
}

// ---------------------------------------------------------------------------------------------
// Random numbers: Philox4x32-10 (Salmon et al. 2011) keyed by the run's seed, counted by
// (trajectory, its jump count, purpose, index).
__device__ uint4 philox(uint4 c,uint2 k){
    #pragma unroll
    for(int i=0;i<10;i++){
        const u32 hi0=__umulhi(0xD2511F53u,c.x),lo0=0xD2511F53u*c.x,hi1=__umulhi(0xCD9E8D57u,c.z),lo1=0xCD9E8D57u*c.z;
        c=make_uint4(hi1^c.y^k.x,lo1,hi0^c.w^k.y,lo0);k.x+=0x9E3779B9u;k.y+=0xBB67AE85u;
    }
    return c;
}
// A double uniform in (0, 1) from two words.
__device__ __forceinline__ double open_uniform(u32 a,u32 b){return (((double)(a>>5))*67108864.0+(double)(b>>6)+0.5)*(1.0/9007199254740992.0);}
#define PURPOSE_THRESHOLD 0u
#define PURPOSE_CHANNEL 1u
#define PURPOSE_DIRECTION 2u
#define PURPOSE_NOISE 3u

// The waiting-time threshold ln r of a trajectory's waiting period after `jumps` jumps.
__device__ double threshold_of(uint2 seed,u32 t,u32 jumps){
    const uint4 r=philox(make_uint4(t,jumps,PURPOSE_THRESHOLD,0u),seed);return log(open_uniform(r.x,r.y));
}
extern "C" __global__ void thresholds(double* threshold,const u32* jumps,int trajectories,uint2 seed){
    const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t<trajectories)threshold[t]=threshold_of(seed,(u32)t,jumps[t]);
}

// Per trajectory state.
struct Trajectories{
    int* components;   // 5 (F = 2), 3 (F = 1), 0 (lost)
    float* scale;      // the renormalization the next kinetic pass applies
    double* log_norm;  // ln of the norm's decay since the last jump
    double* threshold; // ln r of the current waiting period
    u32* jumps;
};

// After the local step: each trajectory's norm from its partial sums. A trajectory whose decay
// reaches its threshold within the step jumps at the step's midpoint (listed in `jumpers`); the
// others are renormalized. One thread per trajectory.
extern "C" __global__ void epilogue(Trajectories tr,int trajectories,const float* partial,int tiles,int* jumpers,int* count){
    const int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=trajectories||tr.components[t]==0)return;
    double n1=0.;for(int i=0;i<tiles;i++)n1+=(double)partial[(u64)t*tiles+i];
    const double ln1=log(n1);
    if(tr.log_norm[t]+ln1>tr.threshold[t]){tr.log_norm[t]+=ln1;tr.scale[t]=(float)rsqrt(n1);}
    else{jumpers[atomicAdd(count,1)]=t;tr.scale[t]=1.f;}
}

// ---------------------------------------------------------------------------------------------
// The jumps. A jump from block `from` scatters a photon from carrier c, the atom going to block
// `to`: L_a = P_to C_a R_c P_from for the photon's polarization along axis a of the frame, with
// R_c = sum_l a_l(r) [R_l - R2_l w_l] to first order in the excited state's kinetic shift
// w_l = recoil_l + hbar k_l.kappa/m (model.rs). The photon's direction n has the distribution
// tr G - n.Re(G).n, G_ab = <L_a psi|L_b psi> summed over the grid: a mixture of linear-dipole
// patterns along the eigenvectors of Re(G). Given n, its polarization is one of two unit vectors
// across n, e, with probability e.Re(G).e. The atom recoils by -hbar k_c n.
//
// Ground states are in block order here: F = 1's three states (rows/columns 0-2), then F = 2's
// five (3-7); a trajectory's component j of block k is block-ordered state (k ? 3 : 0) + j.
struct Jumps{
    // Along the radius, linear between the radii k*step: per near beam l and term (0: R_l,
    // 1: R2_l) the jump matrices C_a R (block-ordered ground row g <- column h) at
    // jump[((((l*2 + term)*3 + a)*8 + g)*8 + h)*radii + k], and the responses R_l (excited e <-
    // block-ordered ground h) at response[((l*24 + e)*8 + h)*radii + k].
    const float2* jump;const float2* response;float step;int radii;
    // Per near beam, 8 floats: carrier, k_x, k_y (rad/um), 1/w^2 (a focused beam's, else 0),
    // its recoil shift hbar k^2/2m (rad/us) and hbar k_x/m, hbar k_y/m (um/us).
    const float* beams;int beam_count;
    const float* cap;        // the absorbing potential's rate per point (1/us)
    const float* wave_number;// per carrier (rad/um)
    int carriers;
    int nx;float dx;         // the grid: point x is at ((x % nx) - nx/2, (x / nx) - ny/2) dx
};
#define EXCITED 24
#define BEAM 8
#define G_SLOTS 25 // Re G_ab (6) per (carrier, to-block), up to 2 carriers x 2 blocks, then the loss rate
#define CHOICE 8   // a jump's choice (jump_choose)

// The plane position (um) of point x.
__device__ __forceinline__ float2 position(int x,int points,int nx,float dx){
    const int ny=points/nx;return make_float2(((x%nx)-nx/2)*dx,ny>1?((x/nx)-ny/2)*dx:0.f);
}
// Beam l's amplitude a_l(r): its plane wave's phase, a focused beam's Gaussian profile.
__device__ __forceinline__ float2 amplitude(const float* b,float2 r,float rho2){
    float s,c;sincosf(b[1]*r.x+b[2]*r.y,&s,&c);const float g=b[3]>0.f?__expf(-rho2*b[3]):1.f;return make_float2(c*g,s*g);
}
// A radial table's entry at sample k + f.
__device__ __forceinline__ float2 radial_entry(const float2* table,int radii,int entry,int k,float f){
    const float2* p=table+(u64)entry*radii+k;const float2 a=__ldg(p),b=__ldg(p+1);
    return make_float2(a.x+f*(b.x-a.x),a.y+f*(b.y-a.y));
}
struct Radius{int k;float f;};
__device__ __forceinline__ Radius radius(const Jumps& J,float rho){
    const float x=rho/J.step;const int k=min((int)x,J.radii-2);return Radius{k,x-k};
}
// out[a][g] = (C_a R_c psi)_g with its first-order terms, from the N
// components of block-ordered columns H.. of psi, kappa_x psi, kappa_y psi at r.
template<int N,int H>
__device__ void emitted(const Jumps& J,int c,float2 r,float rho2,Radius q,const float2 (&psi)[5],const float2 (&dx)[5],const float2 (&dy)[5],float2 (&out)[3][8]){
    #pragma unroll
    for(int a=0;a<3;a++){
        #pragma unroll
        for(int g=0;g<8;g++)out[a][g]=make_float2(0.f,0.f);
    }
    for(int l=0;l<J.beam_count;l++){
        const float* b=J.beams+l*BEAM;if((int)b[0]!=c)continue;
        const float2 amp=amplitude(b,r,rho2);
        float2 s[2][N];
        #pragma unroll
        for(int j=0;j<N;j++){
            s[0][j]=cmul(amp,psi[j]);
            s[1][j]=cscale(cmul(amp,cadd(cscale(psi[j],b[4]),cadd(cscale(dx[j],b[5]),cscale(dy[j],b[6])))),-1.f);
        }
        #pragma unroll
        for(int term=0;term<2;term++){
            #pragma unroll
            for(int a=0;a<3;a++){
                #pragma unroll
                for(int g=0;g<8;g++){
                    float2 z=out[a][g];
                    #pragma unroll
                    for(int j=0;j<N;j++)z=cadd(z,cmul(radial_entry(J.jump,J.radii,(((l*2+term)*3+a)*8+g)*8+H+j,q.k,q.f),s[term][j]));
                    out[a][g]=z;
                }
            }
        }
    }
}

// Moves each jumper from the step's end back to its midpoint: psi <- V_from(-dt/2) psi. Grid
// (tiles, any): the blocks stride over the jumpers.
extern "C" __global__ void __launch_bounds__(256) jump_back(float2* state,int points,const int* components,const int* jumpers,const int* count,Local back){
    const int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=points)return;
    for(int s=blockIdx.y;s<*count;s+=gridDim.y){const int t=jumpers[s];apply_local(state+(u64)t*COMPONENTS*points+x,points,components[t],back,x);}
}

// One point's contribution to a jumper's G matrices (acc, G_SLOTS).
template<int N,int H>
__device__ void rates_point(const Jumps& J,float2 r,float rho2,const float2 (&psi)[5],const float2 (&dx)[5],const float2 (&dy)[5],float* acc){
    const Radius q=radius(J,sqrtf(rho2));
    for(int c=0;c<J.carriers;c++){
        float2 f[3][8];emitted<N,H>(J,c,r,rho2,q,psi,dx,dy,f);
        #pragma unroll
        for(int to=0;to<2;to++){
            float g0=0.f,g1=0.f,g2=0.f,g3=0.f,g4=0.f,g5=0.f;
            #pragma unroll
            for(int i=(to?3:0);i<(to?8:3);i++){
                const float2 a=f[0][i],b=f[1][i],e=f[2][i];
                g0+=norm2(a);g1+=norm2(b);g2+=norm2(e);g3+=a.x*b.x+a.y*b.y;g4+=a.x*e.x+a.y*e.y;g5+=b.x*e.x+b.y*e.y;
            }
            float* g=acc+(c*2+to)*6;g[0]+=g0;g[1]+=g1;g[2]+=g2;g[3]+=g3;g[4]+=g4;g[5]+=g5;
        }
    }
}
// Each jumper's G matrices and loss rate, summed over the block's points into rates[s][tile][25].
extern "C" __global__ void __launch_bounds__(256) jump_rates(const float2* state,const float2* dxs,const float2* dys,int points,const int* components,
        const int* jumpers,const int* count,Jumps J,float* rates){
    __shared__ float scratch[32];
    const int x=blockIdx.x*blockDim.x+threadIdx.x;const bool ok=x<points;
    const float2 r=position(ok?x:0,points,J.nx,J.dx);const float rho2=r.x*r.x+r.y*r.y;
    for(int s=blockIdx.y;s<*count;s+=gridDim.y){
        const int t=jumpers[s],n=components[t];
        float acc[G_SLOTS];
        #pragma unroll
        for(int i=0;i<G_SLOTS;i++)acc[i]=0.f;
        if(ok){
            const u64 base=(u64)t*COMPONENTS*points+x;float2 psi[5],dx[5],dy[5];float w=0.f;
            #pragma unroll
            for(int j=0;j<5;j++){if(j<n){psi[j]=state[base+(u64)j*points];dx[j]=dxs[base+(u64)j*points];dy[j]=dys[base+(u64)j*points];w+=norm2(psi[j]);}else psi[j]=dx[j]=dy[j]=make_float2(0.f,0.f);}
            acc[G_SLOTS-1]=w*__ldg(J.cap+x);
            if(n==5)rates_point<5,3>(J,r,rho2,psi,dx,dy,acc);else rates_point<3,0>(J,r,rho2,psi,dx,dy,acc);
        }
        float* out=rates+((u64)s*gridDim.x+blockIdx.x)*G_SLOTS;
        #pragma unroll
        for(int i=0;i<G_SLOTS;i++){const float v=block_sum(acc[i],scratch);if(threadIdx.x==0)out[i]=v;}
    }
}

// The chosen jump at one point: out = e^{-i k n.r} (sum_a e_a f_a) / |e.L psi| on the to-block.
template<int N,int H,int TO>
__device__ void apply_point(const Jumps& J,int c,float2 r,float rho2,const float2 (&psi)[5],const float2 (&dx)[5],const float2 (&dy)[5],const float* ch,float2 (&out)[5]){
    float2 f[3][8];emitted<N,H>(J,c,r,rho2,radius(J,sqrtf(rho2)),psi,dx,dy,f);
    float sk,ck;sincosf(-__ldg(J.wave_number+c)*(ch[2]*r.x+ch[3]*r.y),&sk,&ck);
    const float2 kick=make_float2(ck*ch[7],sk*ch[7]);
    #pragma unroll
    for(int i=0;i<5;i++){
        if(i>=(TO?5:3)){out[i]=make_float2(0.f,0.f);continue;}
        const int g=(TO?3:0)+i;
        out[i]=cmul(cadd(cadd(cscale(f[0][g],ch[4]),cscale(f[1][g],ch[5])),cscale(f[2][g],ch[6])),kick);
    }
}
// Applies each jumper's chosen jump at its points and evolves the half step to the step's end,
// psi <- V_to(dt/2) psi'; the norm after it to partial.
extern "C" __global__ void __launch_bounds__(256) jump_apply(float2* state,const float2* dxs,const float2* dys,int points,
        const int* components,const int* jumpers,const int* count,Jumps J,const float* choices,Local forward,float* partial){
    __shared__ float scratch[32];
    const int x=blockIdx.x*blockDim.x+threadIdx.x;const bool ok=x<points;
    const float2 r=position(ok?x:0,points,J.nx,J.dx);const float rho2=r.x*r.x+r.y*r.y;
    for(int s=blockIdx.y;s<*count;s+=gridDim.y){
        const int t=jumpers[s];const float* ch=choices+(u64)t*CHOICE;const int to=(int)ch[1];
        if(to<0)continue;
        const int n=components[t],c=(int)ch[0],nt=to==1?5:3;
        float n2=0.f;
        if(ok){
            const u64 base=(u64)t*COMPONENTS*points+x;float2 psi[5],dx[5],dy[5],out[5];
            #pragma unroll
            for(int j=0;j<5;j++){if(j<n){psi[j]=state[base+(u64)j*points];dx[j]=dxs[base+(u64)j*points];dy[j]=dys[base+(u64)j*points];}else psi[j]=dx[j]=dy[j]=make_float2(0.f,0.f);}
            if(n==5){if(to==1)apply_point<5,3,1>(J,c,r,rho2,psi,dx,dy,ch,out);else apply_point<5,3,0>(J,c,r,rho2,psi,dx,dy,ch,out);}
            else{if(to==1)apply_point<3,0,1>(J,c,r,rho2,psi,dx,dy,ch,out);else apply_point<3,0,0>(J,c,r,rho2,psi,dx,dy,ch,out);}
            #pragma unroll
            for(int j=0;j<5;j++)if(j<nt)state[base+(u64)j*points]=out[j];
            n2=apply_local(state+base,points,nt,forward,x);
        }
        n2=block_sum(n2,scratch);
        if(threadIdx.x==0)partial[(u64)t*gridDim.x+blockIdx.x]=n2;
    }
}

// A real symmetric 3 x 3 matrix's eigen-decomposition by cyclic Jacobi rotations: a becomes
// diagonal (the eigenvalues), the columns of v the eigenvectors.
__device__ void jacobi3(double a[3][3],double v[3][3]){
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)v[i][j]=i==j?1.:0.;
    for(int sweep=0;sweep<50;sweep++){
        const double off=a[0][1]*a[0][1]+a[0][2]*a[0][2]+a[1][2]*a[1][2];
        if(off<=1e-30*(a[0][0]*a[0][0]+a[1][1]*a[1][1]+a[2][2]*a[2][2])+1e-300)break;
        for(int p=0;p<2;p++)for(int q=p+1;q<3;q++){
            if(a[p][q]==0.)continue;
            const double theta=(a[q][q]-a[p][p])/(2.*a[p][q]);
            const double t=(theta>=0.?1.:-1.)/(fabs(theta)+sqrt(theta*theta+1.)),c=1./sqrt(t*t+1.),s=t*c;
            for(int k=0;k<3;k++){const double x=a[k][p],y=a[k][q];a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
            for(int k=0;k<3;k++){const double x=a[p][k],y=a[q][k];a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}
            for(int k=0;k<3;k++){const double x=v[k][p],y=v[k][q];v[k][p]=c*x-s*y;v[k][q]=s*x+c*y;}
        }
    }
}
// Two unit vectors completing u to a right-handed orthonormal frame.
__device__ void perpendicular(const double u[3],double p[3],double q[3]){
    const int i=fabs(u[0])<=fabs(u[1])?(fabs(u[0])<=fabs(u[2])?0:2):(fabs(u[1])<=fabs(u[2])?1:2);
    double e[3]={0.,0.,0.};e[i]=1.;
    double d=u[0]*e[0]+u[1]*e[1]+u[2]*e[2];for(int k=0;k<3;k++)p[k]=e[k]-d*u[k];
    d=rsqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);for(int k=0;k<3;k++)p[k]*=d;
    q[0]=u[1]*p[2]-u[2]*p[1];q[1]=u[2]*p[0]-u[0]*p[2];q[2]=u[0]*p[1]-u[1]*p[0];
}
// The choice of a jump: carrier, to-block (-1: lost to the absorbing boundary), the photon's plane
// components n_x, n_y, its polarization e and 1/|e.L psi|. One thread per jumper.
extern "C" __global__ void jump_choose(Trajectories tr,const int* jumpers,const int* count,const float* rates,int tiles,Jumps J,uint2 seed,float* choices){
    const int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=*count)return;
    const int t=jumpers[s];
    double g[G_SLOTS];for(int i=0;i<G_SLOTS;i++){double v=0.;for(int b=0;b<tiles;b++)v+=(double)rates[((u64)s*tiles+b)*G_SLOTS+i];g[i]=v;}
    double total=g[G_SLOTS-1];for(int c=0;c<J.carriers;c++)for(int to=0;to<2;to++){const double* m=g+(c*2+to)*6;total+=m[0]+m[1]+m[2];}
    const uint4 r1=philox(make_uint4((u32)t,tr.jumps[t],PURPOSE_CHANNEL,0u),seed),r2=philox(make_uint4((u32)t,tr.jumps[t],PURPOSE_DIRECTION,0u),seed),
                r3=philox(make_uint4((u32)t,tr.jumps[t],PURPOSE_DIRECTION,1u),seed);
    float* out=choices+(u64)t*CHOICE;
    double u=open_uniform(r1.x,r1.y)*total;
    int pick=-1;
    for(int c=0;c<J.carriers&&pick<0;c++)for(int to=0;to<2&&pick<0;to++){const double* m=g+(c*2+to)*6;const double w=m[0]+m[1]+m[2];if(u<w)pick=c*2+to;else u-=w;}
    if(pick<0){out[0]=0.f;out[1]=-1.f;return;}
    const double* m=g+pick*6;
    double a[3][3]={{m[0],m[3],m[4]},{m[3],m[1],m[5]},{m[4],m[5],m[2]}},v[3][3];
    jacobi3(a,v);
    double w[3];for(int k=0;k<3;k++)w[k]=fmax(a[k][k],0.);
    const double x=open_uniform(r1.z,r1.w)*(w[0]+w[1]+w[2]);const int axis=x<w[0]?0:x<w[0]+w[1]?1:2;
    const double e[3]={v[0][axis],v[1][axis],v[2][axis]};double p[3],q[3];perpendicular(e,p,q);
    // The cosine to the dipole, density (3/4)(1 - u^2): u^3 - 3u + 2t = 0 with t = 2F - 1.
    const double tt=2.*open_uniform(r2.x,r2.y)-1.,cu=2.*cos((2.*PI-acos(-tt))/3.),su=sqrt(fmax(0.,1.-cu*cu));
    double sphi,cphi;sincospi(2.*open_uniform(r2.z,r2.w),&sphi,&cphi);
    double n[3];for(int k=0;k<3;k++)n[k]=cu*e[k]+su*(cphi*p[k]+sphi*q[k]);
    double e1[3],e2[3];perpendicular(n,e1,e2);
    double p1=0.,p2=0.;
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){const double mij=i==j?m[i]:m[2+i+j];p1+=e1[i]*mij*e1[j];p2+=e2[i]*mij*e2[j];}
    p1=fmax(p1,0.);p2=fmax(p2,0.);
    const bool second=open_uniform(r3.x,r3.y)*(p1+p2)>=p1;const double* eps=second?e2:e1;
    out[0]=(float)(pick/2);out[1]=(float)(pick%2);out[2]=(float)n[0];out[3]=(float)n[1];
    out[4]=(float)eps[0];out[5]=(float)eps[1];out[6]=(float)eps[2];out[7]=(float)rsqrt(fmax(second?p2:p1,1e-300));
}

// Each jumper's new waiting period: its block, norm and threshold; the lost are marked.
extern "C" __global__ void jump_finish(Trajectories tr,const int* jumpers,const int* count,const float* choices,const float* partial,int tiles,uint2 seed){
    const int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=*count)return;
    const int t=jumpers[s];const float* ch=choices+(u64)t*CHOICE;
    tr.jumps[t]+=1;
    if(ch[1]<0.f){tr.components[t]=0;return;}
    tr.components[t]=ch[1]>0.f?5:3;
    double n1=0.;for(int i=0;i<tiles;i++)n1+=(double)partial[(u64)t*tiles+i];
    tr.log_norm[t]=log(n1);tr.scale[t]=(float)rsqrt(n1);
    tr.threshold[t]=threshold_of(seed,(u32)t,tr.jumps[t]);
}

// ---------------------------------------------------------------------------------------------
// An output time's observables at the points, added to the trajectories' records: the
// populations (bare ground rho - {Y, rho}/2 with Y = sum_c R_c^dag R_c, by block-ordered state;
// excited sum_c |R_c psi|^2), the position moments and tails; with a density buffer,
// sum_t w_t |psi_t(x)|^2. Grid (tiles, trajectories/SUB): the tables are read once per SUB
// trajectories.
template<int N,int H>
__device__ void observe_point(const Jumps& J,float2 r,float rho2,const float2 (&psi)[5],float* acc){
    const Radius q=radius(J,sqrtf(rho2));
    float2 y[N];
    #pragma unroll
    for(int j=0;j<N;j++)y[j]=make_float2(0.f,0.f);
    for(int c=0;c<J.carriers;c++){
        float2 e[EXCITED];
        #pragma unroll
        for(int i=0;i<EXCITED;i++)e[i]=make_float2(0.f,0.f);
        for(int l=0;l<J.beam_count;l++){
            const float* b=J.beams+l*BEAM;if((int)b[0]!=c)continue;
            const float2 amp=amplitude(b,r,rho2);float2 s[N];
            #pragma unroll
            for(int j=0;j<N;j++)s[j]=cmul(amp,psi[j]);
            #pragma unroll
            for(int i=0;i<EXCITED;i++){
                float2 z=e[i];
                #pragma unroll
                for(int j=0;j<N;j++)z=cadd(z,cmul(radial_entry(J.response,J.radii,(l*EXCITED+i)*8+H+j,q.k,q.f),s[j]));
                e[i]=z;
            }
        }
        #pragma unroll
        for(int i=0;i<EXCITED;i++)acc[REC_EXCITED+i]+=norm2(e[i]);
        for(int l=0;l<J.beam_count;l++){
            const float* b=J.beams+l*BEAM;if((int)b[0]!=c)continue;
            const float2 amp=amplitude(b,r,rho2);
            #pragma unroll
            for(int j=0;j<N;j++){
                float2 z=make_float2(0.f,0.f);
                #pragma unroll
                for(int i=0;i<EXCITED;i++)z=cadd(z,cmulc(e[i],radial_entry(J.response,J.radii,(l*EXCITED+i)*8+H+j,q.k,q.f)));
                y[j]=cadd(y[j],cmulc(z,amp));
            }
        }
    }
    #pragma unroll
    for(int j=0;j<N;j++)acc[REC_GROUND+H+j]=norm2(psi[j])-(psi[j].x*y[j].x+psi[j].y*y[j].y);
}
extern "C" __global__ void __launch_bounds__(256) observe_points(const float2* state,int points,int trajectories,const int* components,
        Jumps J,float tail_x,float tail_y,float* records,const float* weights,float* density){
    __shared__ float scratch[32];
    const int x=blockIdx.x*blockDim.x+threadIdx.x;const bool ok=x<points;
    const float2 r=position(ok?x:0,points,J.nx,J.dx);const float rho2=r.x*r.x+r.y*r.y;
    for(int s=0;s<SUB;s++){
        const int t=blockIdx.y*SUB+s;if(t>=trajectories)break;
        const int n=components[t];if(n==0)continue;
        float acc[REC_KX];
        #pragma unroll
        for(int i=0;i<REC_KX;i++)acc[i]=0.f;
        float tx=0.f,ty=0.f;
        if(ok){
            const u64 base=(u64)t*COMPONENTS*points+x;float2 psi[5];float nn=0.f;
            #pragma unroll
            for(int j=0;j<5;j++){psi[j]=j<n?state[base+(u64)j*points]:make_float2(0.f,0.f);nn+=norm2(psi[j]);}
            if(n==5)observe_point<5,3>(J,r,rho2,psi,acc);else observe_point<3,0>(J,r,rho2,psi,acc);
            acc[REC_NORM]=nn;acc[REC_X]=nn*r.x;acc[REC_X+1]=nn*r.y;acc[REC_X+2]=nn*r.x*r.x;acc[REC_X+3]=nn*r.y*r.y;
            if(fabsf(r.x)>tail_x)tx=nn*r.x*r.x;
            if(fabsf(r.y)>tail_y)ty=nn*r.y*r.y;
            if(density)atomicAdd(density+x,weights[t]*nn);
        }
        float* rec=records+(u64)t*RECORD;
        #pragma unroll
        for(int i=0;i<REC_KX;i++){const float v=block_sum(acc[i],scratch);if(threadIdx.x==0&&v!=0.f)atomicAdd(rec+i,v);}
        tx=block_sum(tx,scratch);ty=block_sum(ty,scratch);
        if(threadIdx.x==0){if(tx!=0.f)atomicAdd(rec+REC_TAIL,tx);if(ty!=0.f)atomicAdd(rec+REC_TAIL+1,ty);}
    }
}

// ---------------------------------------------------------------------------------------------
// Thermal initial states (random-phase thermal pure states): plane 0 of every trajectory gets
// white noise of random phases, then is filtered in imaginary time by its potential
// (`thermal_potential`: one real factor per point) and by the kinetic pass with real tables.
extern "C" __global__ void thermal_noise(float2* state,int points,int trajectories,uint2 seed){
    const int x=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(x>=points||t>=trajectories)return;
    const uint4 r=philox(make_uint4((u32)t,(u32)x,PURPOSE_NOISE,0u),seed);float s,c;sincospif(2.f*(float)open_uniform(r.x,r.y),&s,&c);
    state[(u64)t*COMPONENTS*points+x]=make_float2(c,s);
}
extern "C" __global__ void thermal_potential(float2* state,int points,int trajectories,const float* factors,const int* potential){
    const int x=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(x>=points||t>=trajectories)return;
    float2* p=state+(u64)t*COMPONENTS*points+x;*p=cscale(*p,__ldg(factors+(u64)potential[t]*points+x));
}
// |plane 0|^2 per trajectory as partial sums per tile; grid (tiles, trajectories).
extern "C" __global__ void plane_norms(const float2* state,int points,float* partial){
    __shared__ float scratch[32];
    const int x=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    const float v=block_sum(x<points?norm2(state[(u64)t*COMPONENTS*points+x]):0.f,scratch);
    if(threadIdx.x==0)partial[(u64)t*gridDim.x+blockIdx.x]=v;
}
// psi_c = phi_c chi / |chi| for the trajectory's internal state phi (5 complex per trajectory).
extern "C" __global__ void thermal_expand(float2* state,int points,int trajectories,const int* components,const float2* internal,const float* inverse_norm){
    const int x=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;if(x>=points||t>=trajectories)return;
    float2* p=state+(u64)t*COMPONENTS*points+x;const float2 chi=cscale(p[0],inverse_norm[t]);
    for(int c=components[t]-1;c>=0;c--)p[(u64)c*points]=cmul(internal[t*5+c],chi);
}
