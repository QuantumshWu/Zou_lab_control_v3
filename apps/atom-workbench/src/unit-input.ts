// The v3 unit rule (ui-design.md §0.4; apps/desktop/ui/src/views/device_control.ts): a displayed
// quantity is converted when its unit changes -- the quantity stays, the digits follow the unit;
// a number the operator wrote keeps its digits -- the unit really changes, and so does the quantity.

/** The digits an input's number shows. A quantity kept in a stored unit and read in the operator's
 * carries the float dust of that division (15.7 ns kept in μs reads back 15.699999999999998); the
 * v3 spin box shows such a number by its twelve significant digits when they agree with its
 * fourteen (spin.ts approximateText), and those are the digits a unit change keeps. A number with
 * more digits than that is kept whole. */
export function typedDigits(value:number):number{
  if(!Number.isFinite(value))return value;
  const twelve=Number(value.toPrecision(12));
  return twelve===Number(value.toPrecision(14))?twelve:value;
}

/** A number in a unit: what an input holds, or a display shows. */
export type Reading<U extends string=string>={value:number;unit:U};
