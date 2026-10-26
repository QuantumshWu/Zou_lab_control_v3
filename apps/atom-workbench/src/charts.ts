import {combinedLevelsChart} from './combined-level-chart.js';
import type {SpectralLine} from './types.js';
import {formatNumber} from './format.js';

export const escapeHtml = (value: unknown): string => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
/** A number with fixed decimals (a table column), in the one spelling of format.ts. */
export function number(value:number,places=3):string {return formatNumber(value,{places});}

/** Fixed physical probability scale, shared by chart rendering and live patches.
 * Its lightest/darkest colors keep the same dark text readable at every p. */
export function populationAppearance(population:number):{fill:string;ink:string}{
  const t=Math.max(0,Math.min(1,population));
  const rgb=[Math.round(247-113*t),Math.round(249-70*t),Math.round(251-35*t)];
  return {fill:`rgb(${rgb.join(',')})`,ink:'#323130'};
}

export interface LevelChartOptions {
  line:SpectralLine; selectedIds:string[]; initialId:string;
  populations?:Record<string,number>; populationLabel?:string;
  /** Only a verified long-time result; never reuse a finite-time population. */
  steadyPopulations?:Record<string,number>|null;
  steadyPopulationLabel?:'mean'|'limit';
  /** Actual SVG viewport width, preserving readable text in narrow panels. */
  viewportWidth?:number;
}
export const levelsChart=combinedLevelsChart;

export { populationChart, populationColor,  populationValue, populationLayout, populationSamplingInfo, bindPopulationChart } from './population-chart.js';
export type { PopulationPlot, PopulationChartOptions } from './population-chart.js';
