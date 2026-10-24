import {lineMesh,triangleMesh,rasterizeDepth,type DepthPoint,type DepthTriangle} from './space-raster.js';
/** Hardware depth buffer, with the same per-pixel rule in the Canvas fallback. */
export class SpaceDepthRenderer {
 private gl:WebGL2RenderingContext|null=null;private ctx:CanvasRenderingContext2D|null=null;
 private program?:WebGLProgram;private buffer?:WebGLBuffer;private position=0;private color=0;
 private width=1;private height=1;private ratio=1;private distance=7.6;private triangles:DepthTriangle[]=[];
 constructor(readonly canvas:HTMLCanvasElement){
  try{this.gl=canvas.getContext('webgl2',{alpha:false,antialias:true});}catch{}
  if(this.gl){const gl=this.gl;
   const shader=(type:number,source:string)=>{const s=gl.createShader(type)!;gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s)??'Depth shader failed');return s;};
   const p=gl.createProgram()!;gl.attachShader(p,shader(gl.VERTEX_SHADER,'#version 300 es\nin vec4 position;in vec4 color;out vec4 ink;void main(){gl_Position=position;ink=color;}'));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;in vec4 ink;out vec4 fragment;void main(){fragment=ink;}'));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error('Depth renderer link failed');
   this.program=p;this.buffer=gl.createBuffer()!;this.position=gl.getAttribLocation(p,'position');this.color=gl.getAttribLocation(p,'color');
  }else{this.ctx=canvas.getContext('2d');if(!this.ctx)throw Error('3D depth renderer unavailable');}
 }
 begin(width:number,height:number,dpr:number,distance:number):void{
  this.width=width;this.height=height;this.distance=distance;this.triangles=[];this.ratio=this.gl?dpr:Math.max(1,Math.min(2,dpr));
  const w=Math.round(width*this.ratio),h=Math.round(height*this.ratio);if(this.canvas.width!==w)this.canvas.width=w;if(this.canvas.height!==h)this.canvas.height=h;
 }
 line(a:DepthPoint,b:DepthPoint,width:number,color:string,alpha=1):void{this.triangles.push(...lineMesh(a,b,width,color,alpha));}
 face(points:DepthPoint[],color:string,alpha:number,opaque=false):void{this.triangles.push(...triangleMesh(points,color,alpha,opaque));}
 circle(p:DepthPoint,radius:number,color:string,alpha=1):void{this.face(Array.from({length:40},(_,i)=>({x:p.x+radius*Math.cos(i*Math.PI/20),y:p.y+radius*Math.sin(i*Math.PI/20),z:p.z})),color,alpha,true);}
 flush():void{
  const gl=this.gl;if(!gl){const ratio=this.ratio,triangles=this.triangles.map(t=>({...t,points:t.points.map(p=>({x:p.x*ratio,y:p.y*ratio,z:p.z}))as[DepthPoint,DepthPoint,DepthPoint]})),result=rasterizeDepth(triangles,this.canvas.width,this.canvas.height,this.distance);const frame=this.ctx!.createImageData(this.canvas.width,this.canvas.height);frame.data.set(result.pixels);this.ctx!.putImageData(frame,0,0);return;}
  gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(1,1,1,1);gl.clearDepth(1);gl.depthMask(true);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);gl.useProgram(this.program!);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer!);
  gl.enableVertexAttribArray(this.position);gl.vertexAttribPointer(this.position,4,gl.FLOAT,false,32,0);gl.enableVertexAttribArray(this.color);gl.vertexAttribPointer(this.color,4,gl.FLOAT,false,32,16);
  const draw=(triangles:DepthTriangle[])=>{const data=new Float32Array(triangles.length*24);let at=0;for(const t of triangles)for(const p of t.points){const w=this.distance-p.z,near=.1,far=40,depth=far/(far-near)-far*near/((far-near)*w);data.set([(2*p.x/this.width-1)*w,(1-2*p.y/this.height)*w,(2*depth-1)*w,w,...t.color,t.alpha],at);at+=8;}gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);gl.drawArrays(gl.TRIANGLES,0,data.length/8);};
  gl.disable(gl.BLEND);draw(this.triangles.filter(t=>t.opaque));
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);draw(this.triangles.filter(t=>!t.opaque).sort((a,b)=>a.points.reduce((s,p)=>s+p.z,0)-b.points.reduce((s,p)=>s+p.z,0)));gl.depthMask(true);
 }
 destroy():void{if(this.gl){this.gl.deleteBuffer(this.buffer!);this.gl.deleteProgram(this.program!);}}
}
