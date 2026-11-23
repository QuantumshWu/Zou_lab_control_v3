// The string literals of a Rust source's shipped code, for the tests that read what the solver can say
// (motion.test.mjs): comments and every item under #[cfg(test)] or #[cfg(feature = "prof")] left out (the
// tests' and the profiling feature's, prof.rs: no released build has either), each literal with the calls
// and macros it stands in, innermost last (`Err`, `format!`, `expect`, …); and the modules the shipped code
// declares, so that a crate's shipped files are those its root reaches (`rustModules`).
const CHAR=/^'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\])'/;
const OPEN='([{',CLOSE=')]}';

/** The source as tokens: identifiers, string literals (their text as written) and single punctuation marks. */
function tokens(source){
  const out=[];
  for(let i=0;i<source.length;i++){
    const c=source[i];
    if(/\s/.test(c))continue;
    if(source.startsWith('//',i)){i=source.indexOf('\n',i);if(i<0)break;continue;}
    if(source.startsWith('/*',i)){i=source.indexOf('*/',i+2)+1;continue;}
    // A character, or a lifetime's mark.
    if(c==="'"){const char=CHAR.exec(source.slice(i,i+16));if(char)i+=char[0].length-1;continue;}
    const raw=/^b?r(#*)"/.exec(source.slice(i,i+12));
    if(raw){const start=i+raw[0].length,end=source.indexOf('"'+raw[1],start);out.push({string:source.slice(start,end)});i=end+raw[1].length;continue;}
    if(c==='"'){let j=i+1;while(source[j]!=='"')j+=source[j]==='\\'?2:1;out.push({string:source.slice(i+1,j)});i=j;continue;}
    const word=/^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(i,i+80));
    if(word){out.push({word:word[0]});i+=word[0].length-1;continue;}
    out.push({mark:c});
  }
  return out;
}

/** The source's tokens of its shipped code: every item under #[cfg(test)] or #[cfg(feature = "prof")] left out -- to
 * its `;`, or through its block. */
function shipped(source){
  const list=tokens(source),out=[];
  for(let i=0;i<list.length;i++){
    const t=list[i];
    const cfg=t.mark==='#'&&list[i+1]?.mark==='['&&list[i+2]?.word==='cfg'&&list[i+3]?.mark==='(';
    const marks=cfg&&list[i+4]?.word==='test'&&list[i+5]?.mark===')'&&list[i+6]?.mark===']'?7
      :cfg&&list[i+4]?.word==='feature'&&list[i+5]?.mark==='='&&list[i+6]?.string==='prof'&&list[i+7]?.mark===')'&&list[i+8]?.mark===']'?9:0;
    if(marks){
      let depth=0;i+=marks;
      for(;i<list.length;i++){
        const mark=list[i].mark;if(!mark)continue;
        if(mark==='{'&&depth===0){for(let braces=1;braces>0;){const m=list[++i].mark;if(m==='{')braces++;else if(m==='}')braces--;}break;}
        if(OPEN.includes(mark))depth++;else if(CLOSE.includes(mark))depth--;else if(mark===';'&&depth===0)break;
      }
      continue;
    }
    out.push(t);
  }
  return out;
}

/** The modules the shipped code declares from their own files (`mod name;`): `name`, and `path`, its #[path = "…"]
 * where it has one. */
export function rustModules(source){
  const list=shipped(source),out=[];
  for(let i=0;i<list.length;i++){
    if(list[i].word!=='mod'||!list[i+1]?.word||list[i+2]?.mark!==';')continue;
    // Back over `pub`, `pub(crate)` to the attribute before it, if any.
    let j=i-1;
    if(list[j]?.mark===')'&&list[j-2]?.mark==='('&&list[j-3]?.word==='pub')j-=4;else if(list[j]?.word==='pub')j--;
    const path=list[j]?.mark===']'&&list[j-1]?.string!==undefined&&list[j-2]?.mark==='='&&list[j-3]?.word==='path'&&list[j-4]?.mark==='['&&list[j-5]?.mark==='#'?list[j-1].string:undefined;
    out.push({name:list[i+1].word,path});
  }
  return out;
}

/** Every string literal of the shipped code: `text`, and `within`, the names of the calls and macros around it. */
export function rustLiterals(source){
  const list=shipped(source),out=[],stack=[];
  for(let i=0;i<list.length;i++){
    const t=list[i];
    if(t.mark&&OPEN.includes(t.mark)){
      // What is called: `name(`, `name!(`, `.name(`.
      const before=list[i-1],name=before?.word??(before?.mark==='!'&&list[i-2]?.word?list[i-2].word+'!':undefined);
      stack.push(t.mark==='('?name:undefined);continue;
    }
    if(t.mark&&CLOSE.includes(t.mark)){stack.pop();continue;}
    if(t.string!==undefined)out.push({text:t.string,within:stack.filter(Boolean)});
  }
  return out;
}
/** A format string with every `{…}` filled by `fill`. */
export const filled=(text,fill='1')=>text.replace(/\{[^{}]*\}/g,fill);
/** A literal that reads as English: two words of two letters or more with a space between them, the placeholders left out. */
export const english=text=>/[A-Za-z]{2,} +[A-Za-z]{2,}/.test(filled(text,'*'));
/** A literal written in Chinese. */
export const chinese=text=>/[\u4e00-\u9fff]/.test(text);
