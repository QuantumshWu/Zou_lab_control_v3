/** Screen vertices retain camera-space depth. Opaque intersections are
 * resolved per pixel, independent of object/segment submission order. */
export type DepthPoint={x:number;y:number;z:number};
export type DepthTriangle={points:[DepthPoint,DepthPoint,DepthPoint];color:[number,number,number];alpha:number;opaque:boolean};
export function rgb(hex:string):[number,number,number]{const s=hex.slice(1),full=s.length===3?[...s].map(c=>c+c).join(''):s;return [0,2,4].map(i=>parseInt(full.slice(i,i+2),16)/255)as[number,number,number];}
export function lineMesh(a:DepthPoint,b:DepthPoint,width:number,color:string,alpha=1):DepthTriangle[]{
 const length=Math.hypot(b.x-a.x,b.y-a.y);if(length<1e-8)return [];
 const nx=-(b.y-a.y)/length*width/2,ny=(b.x-a.x)/length*width/2;
 const points=[{x:a.x+nx,y:a.y+ny,z:a.z},{x:a.x-nx,y:a.y-ny,z:a.z},{x:b.x+nx,y:b.y+ny,z:b.z},{x:b.x-nx,y:b.y-ny,z:b.z}];
 const tint=rgb(color).map(c=>1-alpha+alpha*c)as[number,number,number];
 return [{points:[points[0]!,points[1]!,points[2]!],color:tint,alpha:1,opaque:true},{points:[points[2]!,points[1]!,points[3]!],color:tint,alpha:1,opaque:true}];
}
export function triangleMesh(points:DepthPoint[],color:string,alpha:number,opaque=false):DepthTriangle[]{
 const base=rgb(color),tint=(opaque?base.map(c=>1-alpha+alpha*c):base)as[number,number,number];
 return Array.from({length:Math.max(0,points.length-2)},(_,i)=>({points:[points[0]!,points[i+1]!,points[i+2]!]as[DepthPoint,DepthPoint,DepthPoint],color:tint,alpha:opaque?1:alpha,opaque}));
}
export function rasterizeDepth(triangles:readonly DepthTriangle[],width:number,height:number,distance:number){
 const pixels=new Uint8ClampedArray(width*height*4);pixels.fill(255);const depths=new Float64Array(width*height);depths.fill(-Infinity);
 const edge=(a:DepthPoint,b:DepthPoint,x:number,y:number)=>(b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);
 const topLeft=(a:DepthPoint,b:DepthPoint)=>a.y<b.y||(a.y===b.y&&a.x>b.x);
 const draw=(triangle:DepthTriangle)=>{
  let [a,b,c]=triangle.points;let area=edge(a,b,c.x,c.y);if(Math.abs(area)<1e-9)return;if(area<0){[b,c]=[c,b];area=-area;}
  const xmin=Math.max(0,Math.floor(Math.min(a.x,b.x,c.x))),xmax=Math.min(width-1,Math.ceil(Math.max(a.x,b.x,c.x))),ymin=Math.max(0,Math.floor(Math.min(a.y,b.y,c.y))),ymax=Math.min(height-1,Math.ceil(Math.max(a.y,b.y,c.y)));
  const invA=1/(distance-a.z),invB=1/(distance-b.z),invC=1/(distance-c.z),tl0=topLeft(b,c),tl1=topLeft(c,a),tl2=topLeft(a,b);
  for(let y=ymin;y<=ymax;y++)for(let x=xmin;x<=xmax;x++){
   const e0=edge(b,c,x+.5,y+.5),e1=edge(c,a,x+.5,y+.5),e2=edge(a,b,x+.5,y+.5);
   if(e0<0||e1<0||e2<0||(e0===0&&!tl0)||(e1===0&&!tl1)||(e2===0&&!tl2))continue;
   const depth=(e0*invA+e1*invB+e2*invC)/area,index=y*width+x;if(depth<depths[index]!-1e-10)continue;
   if(triangle.opaque)depths[index]=depth;
   for(let channel=0;channel<3;channel++){const at=index*4+channel;pixels[at]=triangle.color[channel]!*255*triangle.alpha+pixels[at]!*(1-triangle.alpha);}
  }
 };
 triangles.filter(t=>t.opaque).forEach(draw);
 triangles.filter(t=>!t.opaque).slice().sort((a,b)=>a.points.reduce((s,p)=>s+p.z,0)-b.points.reduce((s,p)=>s+p.z,0)).forEach(draw);
 return {pixels,depths};
}
