// node literals.mjs [--accept] [--all] -- the mechanical part of the structural check (SPEC §0: no magic numbers).
//
// Scans the engine's sources (motion/src/**/*.rs, without tests: `tests.rs` files and everything from a file's
// `#[cfg(test)]` on) for numeric literals, comments and strings removed, and prints:
//   1. the named constants (`const NAME: T = value;`) -- to be read against SPEC §0's allowed constants;
//   2. every other line holding a literal beyond the structural ones (0, 1, 2 and their float forms: indices, halves,
//      doublings, signs) that is not in the baseline `literals.accepted.json` -- the lines a reviewer has not yet
//      read. A line is known by its file and its text (not its number), so moving code does not flag it again.
// `--accept` writes the baseline from the current sources (after the listed lines have been reviewed); `--all` lists
// the accepted lines too. Exit code 1 when unreviewed lines remain.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url)),src=path.resolve(here,'../src'),baselineFile=path.join(here,'literals.accepted.json');
const args=process.argv.slice(2);
const files=[];
(function walk(dir){for(const f of fs.readdirSync(dir,{withFileTypes:true})){const p=path.join(dir,f.name);
  if(f.isDirectory())walk(p);else if(f.name.endsWith('.rs')&&f.name!=='tests.rs')files.push(p);}})(src);

/** A source's code lines with comments, strings and chars blanked, cut at its test module. */
function code(text){
  const cut=text.search(/^#\[cfg\(test\)\]/m),body=cut<0?text:text.slice(0,cut);
  let out='',i=0,depth=0;
  while(i<body.length){const c=body[i],d=body[i+1];
    if(depth){if(c==='/'&&d==='*'){depth++;i+=2;}else if(c==='*'&&d==='/'){depth--;i+=2;}else{out+=c==='\n'?'\n':' ';i++;}}
    else if(c==='/'&&d==='*'){depth=1;i+=2;}
    else if(c==='/'&&d==='/'){while(i<body.length&&body[i]!=='\n')i++;}
    else if(c==='"'){out+='""';i++;while(i<body.length&&body[i]!=='"'){if(body[i]==='\\')i++;if(body[i]==='\n')out+='\n';i++;}i++;}
    else if(c==='r'&&(d==='"'||d==='#')&&/^r#*"/.test(body.slice(i,i+8))){const hashes=body.slice(i+1).match(/^#*/)[0],end=body.indexOf('"'+hashes,i+2+hashes.length);
      out+='""'+'\n'.repeat((body.slice(i,end).match(/\n/g)??[]).length);i=end+1+hashes.length;}
    else if(c==="'"&&/^'(\\.|[^\\'])'/.test(body.slice(i,i+4))){const m=body.slice(i,i+4).match(/^'(\\.|[^\\'])'/)[0];out+="' '";i+=m.length;}
    else{out+=c;i++;}}
  return out.split('\n');
}
const NUMBER=/(?<![A-Za-z_0-9.])(0x[0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)(?:_?(?:f64|f32|u8|u16|u32|u64|usize|i32|i64|isize))?(?![A-Za-z_0-9])/g;
const STRUCTURAL=new Set(['0','1','2','0.0','1.0','2.0','0.','1.','2.','0.5']);
const constants=[],lines=[];
for(const file of files){
  const rel=path.relative(src,file).replaceAll('\\','/'),raw=fs.readFileSync(file,'utf8').split(/\r?\n/),clean=code(raw.join('\n'));
  clean.forEach((text,i)=>{
    const named=text.match(/^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:const|static)\s+([A-Z_0-9]+)\s*:\s*([^=]+)=\s*(.*)$/);
    if(named){constants.push({file:rel,line:i+1,name:named[1],type:named[2].trim(),value:named[3].trim()});return;}
    // tuple and array indices (`.0`, `[3]`) and type lengths (`[f64; 3]`) are structure, not quantities
    const stripped=text.replace(/\.\d+(?![\d.eE])/g,' ').replace(/;\s*\d+\s*\]/g,'; N]').replace(/\[\s*\d+\s*\]/g,'[N]');
    const found=[...stripped.matchAll(NUMBER)].map(m=>m[1].replaceAll('_','')).filter(v=>!STRUCTURAL.has(v));
    if(found.length)lines.push({file:rel,line:i+1,text:raw[i].trim(),literals:found});});
}
const key=l=>`${l.file}|${l.text}`;
if(args.includes('--accept')){
  fs.writeFileSync(baselineFile,JSON.stringify([...new Set(lines.map(key))].sort(),null,1)+'\n');
  console.log(`accepted ${lines.length} lines into ${path.basename(baselineFile)}`);process.exit(0);
}
const accepted=new Set(fs.existsSync(baselineFile)?JSON.parse(fs.readFileSync(baselineFile,'utf8')):[]);
console.log(`# named constants (${constants.length}): read against SPEC §0`);
for(const c of constants)console.log(`${c.file}:${c.line}  ${c.name}: ${c.type} = ${c.value}`);
const fresh=lines.filter(l=>!accepted.has(key(l)));
console.log(`\n# lines with literals beyond 0, 1, 2, 0.5: ${lines.length} (${lines.length-fresh.length} accepted, ${fresh.length} unreviewed)`);
for(const l of args.includes('--all')?lines:fresh)console.log(`${l.file}:${l.line}  [${l.literals.join(', ')}]  ${l.text.slice(0,150)}`);
process.exit(fresh.length?1:0);
