// Mock replies for DOM tests. The compiler itself is tested in zlc-scan and
// through the real Python/native transport; this fixture never expands ranges
// or evaluates linked formulas. Assertions inspect the authored descriptors.
// `facts.link_facts` (by linked port) stands in for the native preview of a formula.
export function scanFacts(facts, axes, source = JSON.stringify({axes})) {
  const rows = axes.map(axis => ({port:axis.port, unit:axis.unit ?? '', values:[...(axis.values ?? [])],
    ...(axis.range ? {range:{...axis.range}} : {}), mode:axis.mode ?? 'range', value_text:axis.value_text ?? '',
    ...(axis.coordinate_of !== undefined ? {coordinate_of:axis.coordinate_of} : {}),
    ...(axis.links || axis.link_enabled ? {link_enabled:axis.link_enabled === true,links:structuredClone(axis.links ?? [])} : {})}));
  const axis_facts = rows.map(axis => {
    const list = axis.value_text.trim() ? axis.value_text.split(',').map(Number) : [];
    const values_bounds = list.length && list.every(Number.isFinite) ? [Math.min(...list),Math.max(...list)] : null;
    const v=axis.values, delta=v.length>1?(v.at(-1)-v[0])/(v.length-1):0;
    const range_uniform = Boolean(axis.range) || v.every((value,index)=>Math.abs(value-v[0]-index*delta)<1e-10);
    const played = axis.mode==='values'?(list.length?{first:list[0],last:list.at(-1),count:list.length}:null)
      :axis.range?{first:axis.range.start,last:axis.range.stop,count:axis.range.count}:v.length?{first:v[0],last:v.at(-1),count:v.length}:null;
    const links = axis.link_enabled && facts.link_facts ? axis.links.map(link => facts.link_facts[link.port] ? {port:link.port,problem:'',...facts.link_facts[link.port]} : null) : [];
    return {port:axis.port,mode:axis.mode,value_text:axis.value_text,range_uniform,values_bounds,played,links,
      active_bounds:axis.mode==='values'?values_bounds:axis.range?[Math.min(axis.range.start,axis.range.stop),Math.max(axis.range.start,axis.range.stop)]:v.length?[Math.min(...v),Math.max(...v)]:null};
  });
  return {...facts,rows,axis_facts,plan_source:source};
}

export const range = axis => axis.range ?? {start:axis.values[0],stop:axis.values.at(-1),count:axis.values.length};
