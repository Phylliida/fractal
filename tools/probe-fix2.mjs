// probe-fix2.mjs — diagnose the texture-Z df64 collapse further. Barriering the
// fetched Z (probe-fix) did NOT help, so it isn't the Z value. Variants:
//   BASE      : getZ(m++) used in the df64 math                      (collapses)
//   FIXED0    : getZ(0) every iter (loop-INVARIANT index) used in math
//   DEADFETCH : const Z in the math; texelFetch(m++) kept live by nudging dc by eps*v
//   PRESPLIT  : pass the Veltkamp split of Z.hi in the texture, skip the in-shader
//               split of the texture value (only dx — an arithmetic value — is split)
import { chromium } from '@playwright/test';
import { resolveChromium } from './chromium-launch.mjs';
import { DF64_LIB } from '../src/gpu/glsl.js';

const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];
const CRE = -0.743643887037158704752191506114774, CIM = 0.131825904205311970493132056385139;

function buildRef(maxIter) {
  const zx = new Float32Array(maxIter + 1), zy = new Float32Array(maxIter + 1);
  const zxlo = new Float32Array(maxIter + 1), zylo = new Float32Array(maxIter + 1);
  let x = 0, y = 0, n = 0;
  for (; n <= maxIter; n++) {
    const hx = Math.fround(x), hy = Math.fround(y);
    zx[n] = hx; zxlo[n] = Math.fround(x - hx); zy[n] = hy; zylo[n] = Math.fround(y - hy);
    const nx = x*x - y*y + CRE, ny = 2*x*y + CIM; x = nx; y = ny;
    if (x*x + y*y > 4) { n++; break; }
  }
  return { zx, zy, zxlo, zylo, len: n - 1 };
}

// A ds_mul whose FIRST operand is pre-split: as=(a_hi, a_lo) supplied, a=(a.x,a.y).
const DF64_PS = `
vec2 ds_mul_ps(vec2 a, vec2 a_hilo, vec2 b){
  const float SPLIT = 4097.0;
  float a_hi = a_hilo.x, a_lo = a_hilo.y;
  float p  = ob(a.x * b.x);
  float cb = ob(SPLIT * b.x); float b_hi = ob(cb - ob(cb - b.x)); float b_lo = ob(b.x - b_hi);
  float e = ob(ob(ob(ob(ob(a_hi*b_hi) - p) + ob(a_hi*b_lo)) + ob(a_lo*b_hi)) + ob(a_lo*b_lo));
  e = ob(e + ob(ob(a.x*b.y) + ob(a.y*b.x)));
  float hi = ob(p + e); float lo = ob(e - ob(hi - p));
  return vec2(hi, lo);
}`;

const FRAGS = {
  BASE: `
  void getZ(int m, out vec2 Zx, out vec2 Zy){ vec4 v=texelFetch(uRef,ivec2(m%uRefW,m/uRefW),0); Zx=v.xy; Zy=v.zw; }
  #define MUL(Zc, Zs, dc) ds_mul(Zc, dc)
  #define ADVANCE m++; getZ(m, Zx, Zy);
  #define INITZ getZ(0, Zx, Zy);`,
  FIXED0: `
  void getZ(int m, out vec2 Zx, out vec2 Zy){ vec4 v=texelFetch(uRef,ivec2(m%uRefW,m/uRefW),0); Zx=v.xy; Zy=v.zw; }
  #define MUL(Zc, Zs, dc) ds_mul(Zc, dc)
  #define ADVANCE m++; getZ(0, Zx, Zy);
  #define INITZ getZ(0, Zx, Zy);`,
};

// BASE / FIXED0 share a template; PRESPLIT + DEADFETCH need their own mains.
function fragBase(getzAdvance, fixedIndex) {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform sampler2D uRef; uniform int uRefW, uK; uniform vec2 uDcx, uDcy;
out vec4 frag;
void getZ(int m, out vec2 Zx, out vec2 Zy){ vec4 v=texelFetch(uRef,ivec2(m%uRefW,m/uRefW),0); Zx=v.xy; Zy=v.zw; }
void main(){
  vec2 dcx=uDcx, dcy=uDcy, dx=ds_set(0.0), dy=ds_set(0.0); int m=0;
  vec2 Zx,Zy; getZ(0,Zx,Zy);
  for (int i=0;i<100000000;i++){ if(i>=uK)break;
    vec2 t1=ds_sub(ds_mul(Zx,dx),ds_mul(Zy,dy));
    vec2 t2=ds_add(ds_mul(Zx,dy),ds_mul(Zy,dx));
    vec2 sx=ds_sub(ds_mul(dx,dx),ds_mul(dy,dy)); vec2 sy=ds_mul(dx,dy);
    dx=ds_add(ds_add(ds_add(t1,t1),sx),dcx); dy=ds_add(ds_add(ds_add(t2,t2),ds_add(sy,sy)),dcy);
    m++; getZ(${fixedIndex ? '0' : 'm'},Zx,Zy);
  }
  frag=vec4(dx.x,dx.y,dy.x,dy.y);
}`;
}

// DEADFETCH: math uses CONST Z; the texelFetch(m) is kept live by adding eps*v to dc.
function fragDead() {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform sampler2D uRef; uniform int uRefW, uK; uniform vec2 uDcx, uDcy;
out vec4 frag;
void main(){
  vec2 Zx=ds_set(0.31), Zy=ds_set(0.43);
  vec2 dx=ds_set(0.0), dy=ds_set(0.0); int m=0;
  for (int i=0;i<100000000;i++){ if(i>=uK)break;
    vec4 v=texelFetch(uRef,ivec2(m%uRefW,m/uRefW),0);
    vec2 dcx=ds_add(uDcx, vec2(v.x*1e-30,0.0));   // keep the fetch live, negligibly
    vec2 dcy=uDcy;
    vec2 t1=ds_sub(ds_mul(Zx,dx),ds_mul(Zy,dy));
    vec2 t2=ds_add(ds_mul(Zx,dy),ds_mul(Zy,dx));
    vec2 sx=ds_sub(ds_mul(dx,dx),ds_mul(dy,dy)); vec2 sy=ds_mul(dx,dy);
    dx=ds_add(ds_add(ds_add(t1,t1),sx),dcx); dy=ds_add(ds_add(ds_add(t2,t2),ds_add(sy,sy)),dcy);
    m++;
  }
  frag=vec4(dx.x,dx.y,dy.x,dy.y);
}`;
}

// PRESPLIT: texture carries (Zx.hi, Zx_split_lo, Zy.hi, Zy_split_lo); we reconstruct
// a_lo for the df64 (lo word) separately is not available, so PRESPLIT here only
// tests whether avoiding the IN-SHADER split of the texture hi word fixes it. We pass
// the df64 lo word in a 2nd texture. (Diagnostic only.)
function fragPresplit() {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
${DF64_PS}
uniform sampler2D uRef;    // (Zx.hi, Zx.lo, Zy.hi, Zy.lo)
uniform sampler2D uRefS;   // (Zx_hi12, Zx_lo12, Zy_hi12, Zy_lo12) Veltkamp split of the hi words
uniform int uRefW, uK; uniform vec2 uDcx, uDcy;
out vec4 frag;
void main(){
  vec2 dcx=uDcx, dcy=uDcy, dx=ds_set(0.0), dy=ds_set(0.0); int m=0;
  vec4 v=texelFetch(uRef,ivec2(0,0),0); vec4 s=texelFetch(uRefS,ivec2(0,0),0);
  vec2 Zx=v.xy, Zy=v.zw; vec2 Zxs=s.xy, Zys=s.zw;
  for (int i=0;i<100000000;i++){ if(i>=uK)break;
    vec2 t1=ds_sub(ds_mul_ps(Zx,Zxs,dx),ds_mul_ps(Zy,Zys,dy));
    vec2 t2=ds_add(ds_mul_ps(Zx,Zxs,dy),ds_mul_ps(Zy,Zys,dx));
    vec2 sx=ds_sub(ds_mul(dx,dx),ds_mul(dy,dy)); vec2 sy=ds_mul(dx,dy);
    dx=ds_add(ds_add(ds_add(t1,t1),sx),dcx); dy=ds_add(ds_add(ds_add(t2,t2),ds_add(sy,sy)),dcy);
    m++;
    int xx=m%uRefW, yy=m/uRefW;
    v=texelFetch(uRef,ivec2(xx,yy),0); s=texelFetch(uRefS,ivec2(xx,yy),0);
    Zx=v.xy; Zy=v.zw; Zxs=s.xy; Zys=s.zw;
  }
  frag=vec4(dx.x,dx.y,dy.x,dy.y);
}`;
}

const RUN = (args) => {
  const { progs, ref, refSplit, refLen, dcx, dcy, Ks } = args;
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position=vec4(p,0.,1.); }`;
  function sh(t,s){ const o=gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error('compile: '+gl.getShaderInfoLog(o)); return o; }
  const RW=2048, RH=Math.ceil((refLen+1)/RW);
  function mkTex(data){ const t=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,RW,RH,0,gl.RGBA,gl.FLOAT,data);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST); return t; }
  const t0=new Float32Array(RW*RH*4), t1=new Float32Array(RW*RH*4);
  for(let i=0;i<=refLen;i++){ t0[i*4]=ref.zx[i]; t0[i*4+1]=ref.zxlo[i]; t0[i*4+2]=ref.zy[i]; t0[i*4+3]=ref.zylo[i];
    t1[i*4]=refSplit.xhi[i]; t1[i*4+1]=refSplit.xlo[i]; t1[i*4+2]=refSplit.yhi[i]; t1[i*4+3]=refSplit.ylo[i]; }
  const rt=mkTex(t0), rs=mkTex(t1);
  const ct=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,ct);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,1,1,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
  const fbo=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,ct,0);
  const buf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  const out={ renderer, res:{} };
  for(const p of progs){
    const prog=gl.createProgram(); gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS)); gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,p.frag));
    gl.bindAttribLocation(prog,0,'p'); gl.linkProgram(prog);
    if(!gl.getProgramParameter(prog,gl.LINK_STATUS)) throw new Error('link '+p.name+': '+gl.getProgramInfoLog(prog));
    gl.useProgram(prog); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,rt); gl.uniform1i(gl.getUniformLocation(prog,'uRef'),0);
    const ls=gl.getUniformLocation(prog,'uRefS'); if(ls){ gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,rs); gl.uniform1i(ls,1); }
    gl.uniform1i(gl.getUniformLocation(prog,'uRefW'),RW);
    const ob=gl.getUniformLocation(prog,'uOptBarrier'); if(ob) gl.uniform1i(ob,0);
    gl.uniform2f(gl.getUniformLocation(prog,'uDcx'),dcx[0],dcx[1]); gl.uniform2f(gl.getUniformLocation(prog,'uDcy'),dcy[0],dcy[1]);
    const uK=gl.getUniformLocation(prog,'uK'); gl.viewport(0,0,1,1); gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
    const r={};
    for(const K of Ks){ gl.uniform1i(uK,K); gl.drawArrays(gl.TRIANGLES,0,3);
      const px=new Float32Array(4); gl.readPixels(0,0,1,1,gl.RGBA,gl.FLOAT,px); r[K]=[px[0],px[1],px[2],px[3]]; }
    out.res[p.name]=r;
  }
  return out;
};

function cpuIter(ref, K, dcxd, dcyd, fixed0) {
  const sp=(h,l)=>h+l; let dx=0,dy=0,m=0;
  for(let i=0;i<K;i++){ const Zx=sp(ref.zx[m],ref.zxlo[m]),Zy=sp(ref.zy[m],ref.zylo[m]);
    const t1=Zx*dx-Zy*dy,t2=Zx*dy+Zy*dx,sx=dx*dx-dy*dy,sy=dx*dy;
    dx=2*t1+sx+dcxd; dy=2*t2+2*sy+dcyd; m=fixed0?0:m+1; }
  return [dx,dy];
}
function df64Split(v){ const hi=Math.fround(v); return [hi,Math.fround(v-hi)]; }
function veltSplit(x){ const SPLIT=4097; const c=Math.fround(SPLIT*x); const hi=Math.fround(c-Math.fround(c-x)); const lo=Math.fround(x-hi); return [hi,lo]; }

async function renderOn(args, payload){
  const browser=await chromium.launch({ executablePath:resolveChromium(), args });
  try{ const page=await browser.newPage(); page.on('pageerror',(e)=>console.error('PAGE ERROR:',e.message));
    await page.goto('about:blank'); return await page.evaluate(RUN, payload); } finally { await browser.close(); }
}

const radius=1.5*2**-50, ref=buildRef(200), scale=2*radius/96;
const dcxd=7*scale, dcyd=-5*scale, dcx=df64Split(dcxd), dcy=df64Split(dcyd);
const refSplit={ xhi:new Float32Array(ref.len+1), xlo:new Float32Array(ref.len+1), yhi:new Float32Array(ref.len+1), ylo:new Float32Array(ref.len+1) };
for(let i=0;i<=ref.len;i++){ const [xh,xl]=veltSplit(ref.zx[i]); const [yh,yl]=veltSplit(ref.zy[i]); refSplit.xhi[i]=xh; refSplit.xlo[i]=xl; refSplit.yhi[i]=yh; refSplit.ylo[i]=yl; }
const Ks=[5,20,100];
const progs=[
  { name:'BASE',      frag:fragBase('',false), fixed0:false },
  { name:'FIXED0',    frag:fragBase('',true),  fixed0:true },
  { name:'DEADFETCH', frag:fragDead(),         fixed0:false },
  { name:'PRESPLIT',  frag:fragPresplit(),     fixed0:false },
];
const payload={ progs, ref, refSplit, refLen:ref.len, dcx, dcy, Ks };
const gpu=await renderOn(VULKAN, payload);
console.log(`renderer: ${gpu.renderer}\n`);
console.log('variant    '+Ks.map((K)=>`K=${K}`.padEnd(20)).join(''));
for(const p of progs){
  const cells=Ks.map((K)=>{ const [odx,ody]=cpuIter(ref,K,dcxd,dcyd,p.fixed0); const mag=Math.hypot(odx,ody)||1;
    const g=gpu.res[p.name][K]; const err=Math.hypot((g[0]+g[1])-odx,(g[2]+g[3])-ody)/mag;
    const v=err<1e-10?'INTACT':(err<1e-4?'PARTIAL':'COLLAPSED'); return `${err.toExponential(1)} [${v}]`.padEnd(20); });
  console.log(p.name.padEnd(10)+' '+cells.join(''));
}
