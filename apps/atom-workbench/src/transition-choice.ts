export type ComparisonChoice={selected:string[];from?:string};
export function comparisonPair(choice:ComparisonChoice,origin:string){return {from:choice.from??(choice.selected.length>1?choice.selected[0]!:origin),to:choice.selected.at(-1)};}
export function chooseComparisonFrom(choice:ComparisonChoice,id:string):ComparisonChoice{return {from:id||undefined,selected:choice.selected.length?[choice.selected.at(-1)!]:[]};}
export function chooseComparisonTo(choice:ComparisonChoice,id:string):ComparisonChoice{return {from:choice.from??(choice.selected.length>1?choice.selected[0]:undefined),selected:id?[id]:[]};}
export function swapComparison(choice:ComparisonChoice,origin:string):ComparisonChoice{const p=comparisonPair(choice,origin);return p.to?{from:p.to,selected:[p.from]}:choice;}
export function clickComparison(choice:ComparisonChoice,id:string):ComparisonChoice{
 if(choice.from){if(choice.from===id)return {selected:choice.selected.filter(s=>s!==id)};return {...choice,selected:choice.selected.at(-1)===id?[]:[id]};}
 return {selected:choice.selected.includes(id)?choice.selected.filter(s=>s!==id):choice.selected.length<2?[...choice.selected,id]:[choice.selected[0]!,id]};
}
