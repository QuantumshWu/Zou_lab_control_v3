// The 2D motion engine's kernels (src/engine.rs drives them). Every kernel is a short pass over the
// trajectories' wavefunctions; nothing loops for long, so no kernel nears the display watchdog.
//
// Layout: trajectory t's wavefunction is `state[t][c][y][x]` (complex float2), c < 5 its internal
// components in its hyperfine block (F = 2: 5, F = 1: 3; a lost trajectory has none). A plane is
// one component's NY x NX grid.

typedef unsigned int u32;
typedef unsigned long long u64;

#define COMPONENTS 5

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
// call both (they shuffle); a team's lanes are 32/T teams apart in no way: xor offsets < T stay
// inside a team. The twiddles are computed once per kernel into registers.
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
// The kinetic step on planes that fit in shared memory: one block per plane, exp(-i p^2 tau/2m)
// as K_x then K_y (they commute). `kx` and `ky` hold the step's factor per frequency position
// (bit-reversed order), including the 1/N of each inverse transform; `scale[t]` renormalizes the
// trajectory as it is read. Planes of lost trajectories and the unused components of F = 1 ones
// are skipped.
template<int NX,int NY>
__device__ void kinetic_plane(float2* __restrict__ plane,float scale,const float2* __restrict__ kx,const float2* __restrict__ ky,float2* sm){
    using FX=Fft<NX>;using FY=Fft<NY>;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=blockDim.x>>5;
    constexpr int S=NX+1; // padded row stride: column reads are free of bank conflicts
    {
        constexpr int teams=32/FX::T;const int t=lane%FX::T,team=lane/FX::T;
        FX f;f.init(t);
        float2 phase[FX::E];
        #pragma unroll
        for(int e=0;e<FX::E;e++)phase[e]=__ldg(kx+e*FX::T+t);
        for(int r0=warp*teams;r0<NY;r0+=warps*teams){
            const int r=r0+team;const bool ok=r<NY;float2 v[FX::E];
            #pragma unroll
            for(int e=0;e<FX::E;e++)v[e]=ok?cscale(plane[r*NX+e*FX::T+t],scale):make_float2(0.f,0.f);
            f.forward(v,t);
            #pragma unroll
            for(int e=0;e<FX::E;e++)v[e]=cmul(v[e],phase[e]);
            f.inverse(v,t);
            if(ok){
                #pragma unroll
                for(int e=0;e<FX::E;e++)sm[r*S+e*FX::T+t]=v[e];
            }
        }
    }
    __syncthreads();
    if(NY>1){
        constexpr int teams=32/FY::T;const int t=lane%FY::T,team=lane/FY::T;
        FY f;f.init(t);
        float2 phase[FY::E];
        #pragma unroll
        for(int e=0;e<FY::E;e++)phase[e]=__ldg(ky+e*FY::T+t);
        for(int c0=warp*teams;c0<NX;c0+=warps*teams){
            const int c=c0+team;const bool ok=c<NX;float2 v[FY::E];
            #pragma unroll
            for(int e=0;e<FY::E;e++)v[e]=ok?sm[(e*FY::T+t)*S+c]:make_float2(0.f,0.f);
            f.forward(v,t);
            #pragma unroll
            for(int e=0;e<FY::E;e++)v[e]=cmul(v[e],phase[e]);
            f.inverse(v,t);
            if(ok){
                #pragma unroll
                for(int e=0;e<FY::E;e++)sm[(e*FY::T+t)*S+c]=v[e];
            }
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<NX*NY;i+=blockDim.x)plane[i]=sm[(i/NX)*S+i%NX];
}

#define KINETIC(NX,NY) \
extern "C" __global__ void __launch_bounds__(256) kinetic_##NX##x##NY(float2* state,const int* __restrict__ components,const float* __restrict__ scale,const float2* __restrict__ kx,const float2* __restrict__ ky){ \
    extern __shared__ float2 sm[]; \
    const int plane=blockIdx.x,t=plane/COMPONENTS,c=plane%COMPONENTS; \
    if(c>=components[t])return; \
    kinetic_plane<NX,NY>(state+(u64)plane*(NX*NY),scale[t],kx,ky,sm); \
}
KINETIC(16,16)
KINETIC(32,32)
KINETIC(64,64)
KINETIC(64,1)
KINETIC(128,1)
KINETIC(256,1)
KINETIC(512,1)

// ---------------------------------------------------------------------------------------------
// The local step psi(x) <- V_F(x) psi(x): at each grid point the trajectory's block's matrix
// (F = 2: `v5`, 5 x 5; F = 1: `v3`, 3 x 3; entry (i, j) of point x at v[(i*n+j)*points + x]).
// Each block takes 256 points of SUB trajectories, so a table entry is read once per SUB of them,
// and writes each trajectory's sum of |psi|^2 over its points to partial[t*tiles + tile].
template<int SUB>
__device__ void local_points(float2* __restrict__ state,int points,int trajectories,const int* __restrict__ components,
                             const float2* __restrict__ v5,const float2* __restrict__ v3,float* __restrict__ partial){
    __shared__ float sums[SUB][8];
    const int x=blockIdx.x*blockDim.x+threadIdx.x,tiles=gridDim.x,t0=blockIdx.y*SUB;
    const bool ok=x<points;
    float2 m[25];
    #pragma unroll
    for(int k=0;k<25;k++)m[k]=ok?__ldg(v5+(u64)k*points+x):make_float2(0.f,0.f);
    #pragma unroll 1
    for(int s=0;s<SUB;s++){
        const int t=t0+s;float n2=0.f;
        if(t<trajectories&&ok){
            float2* psi=state+(u64)t*COMPONENTS*points+x;const int n=components[t];
            if(n==5){
                float2 a[5],b[5];
                #pragma unroll
                for(int i=0;i<5;i++)a[i]=psi[(u64)i*points];
                #pragma unroll
                for(int i=0;i<5;i++){float2 z=make_float2(0.f,0.f);
                    #pragma unroll
                    for(int j=0;j<5;j++)z=cadd(z,cmul(m[i*5+j],a[j]));b[i]=z;n2+=norm2(z);}
                #pragma unroll
                for(int i=0;i<5;i++)psi[(u64)i*points]=b[i];
            }else if(n==3){
                float2 a[3],b[3];
                #pragma unroll
                for(int i=0;i<3;i++)a[i]=psi[(u64)i*points];
                #pragma unroll
                for(int i=0;i<3;i++){float2 z=make_float2(0.f,0.f);
                    #pragma unroll
                    for(int j=0;j<3;j++)z=cadd(z,cmul(__ldg(v3+(u64)(i*3+j)*points+x),a[j]));b[i]=z;n2+=norm2(z);}
                #pragma unroll
                for(int i=0;i<3;i++)psi[(u64)i*points]=b[i];
            }
        }
        #pragma unroll
        for(int o=16;o>0;o/=2)n2+=__shfl_xor_sync(0xffffffffu,n2,o);
        if((threadIdx.x&31)==0)sums[s][threadIdx.x>>5]=n2;
    }
    __syncthreads();
    if(threadIdx.x<SUB&&t0+threadIdx.x<trajectories){
        float total=0.f;for(int w=0;w<(int)(blockDim.x>>5);w++)total+=sums[threadIdx.x][w];
        partial[(u64)(t0+threadIdx.x)*tiles+blockIdx.x]=total;
    }
}
extern "C" __global__ void __launch_bounds__(256) local_step(float2* state,int points,int trajectories,const int* components,const float2* v5,const float2* v3,float* partial){
    local_points<8>(state,points,trajectories,components,v5,v3,partial);
}

// ---------------------------------------------------------------------------------------------
// A probe for the build pipeline: out[i] = a * in[i].
extern "C" __global__ void probe_scale(const float* in,float* out,float a,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a*in[i];
}
