/** Electronic reduced E1 matrix-element magnitudes, in e a0.
 * Arora & Sahoo, PRA 86, 033416 (2012), Tables II, IV and V:
 * https://arxiv.org/pdf/1209.0613 . The optimized 4D values are used.
 * Energies are the NIST values distributed by ARC (eV above 5S):
 * https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/master/arc/data/rb_NIST_level_data.ascii
 * These are a finite valence sum; the Rb+ core is CORE_POLARIZABILITY_AU below.
 * Tails above these states and the continuum are not included.
 * Higher-state hyperfine/isotope shifts are not resolved in this dataset. */
export interface VirtualManifold { name:string; J:number; L:number; energyEV:number; reducedEA0:number }
export const VIRTUAL_MANIFOLDS:Record<'ground'|'D1'|'D2',readonly VirtualManifold[]> = {
  ground:[
    ['6P1/2',.5,1,2.9402953,.342],['6P3/2',1.5,1,2.9499053,.553],
    ['7P1/2',.5,1,3.4511026,.118],['7P3/2',1.5,1,3.4554532,.207],
    ['8P1/2',.5,1,3.6990611,.061],['8P3/2',1.5,1,3.7013982,.114],
    ['9P1/2',.5,1,3.8384156,.046],['9P3/2',1.5,1,3.8398142,.074],
  ].map(row=>({name:row[0] as string,J:row[1] as number,L:row[2] as number,energyEV:row[3] as number,reducedEA0:row[4] as number})),
  D1:[
    ['6S1/2',.5,0,2.4961131,4.144],['7S1/2',.5,0,3.2622024,.962],['8S1/2',.5,0,3.6013462,.507],['9S1/2',.5,0,3.7813979,.333],['10S1/2',.5,0,3.8884335,.235],
    ['4D3/2',1.5,2,2.3997946,8.069],['5D3/2',1.5,2,3.1864604,1.184],['6D3/2',1.5,2,3.5567504,1.002],['7D3/2',1.5,2,3.7542555,.75],['8D3/2',1.5,2,3.8709652,.58],['9D3/2',1.5,2,3.9454072,.45],
  ].map(row=>({name:row[0] as string,J:row[1] as number,L:row[2] as number,energyEV:row[3] as number,reducedEA0:row[4] as number})),
  D2:[
    ['6S1/2',.5,0,2.4961131,6.048],['7S1/2',.5,0,3.2622024,1.363],['8S1/2',.5,0,3.6013462,.714],['9S1/2',.5,0,3.7813979,.468],['10S1/2',.5,0,3.8884335,.330],
    ['4D3/2',1.5,2,2.3997946,3.65],['5D3/2',1.5,2,3.1864604,.59],['6D3/2',1.5,2,3.5567504,.48],['7D3/2',1.5,2,3.7542555,.355],['8D3/2',1.5,2,3.8709652,.272],['9D3/2',1.5,2,3.9454072,.212],
    ['4D5/2',2.5,2,2.3997393,10.89],['5D5/2',2.5,2,3.1868276,1.76],['6D5/2',2.5,2,3.5570310,1.42],['7D5/2',2.5,2,3.7544424,1.06],['8D5/2',2.5,2,3.8710908,.81],['9D5/2',2.5,2,3.9454934,.593],
  ].map(row=>({name:row[0] as string,J:row[1] as number,L:row[2] as number,energyEV:row[3] as number,reducedEA0:row[4] as number})),
};
/** Scalar polarizability of the Rb+ core seen by each 32-state manifold, in
 * atomic units (e^2 a0^2 / Eh): core alpha_c plus core-valence alpha_vc.
 * Arora & Sahoo, PRA 86, 033416 (2012), Tables II, IV and V (static values;
 * the same work and dataset as the matrix elements above):
 *   5s: alpha_c = 9.1(5), alpha_vc = -0.26(2);  5p1/2, 5p3/2: alpha_c = 9.1(5), alpha_vc ~ 0.0.
 * Their 770 nm and 1064 nm tables give alpha_c = 9.2-9.3(5): below 3 eV the
 * frequency dependence is within the stated uncertainty, so the static value
 * is used for every laser. The core has no vector or tensor part. The same
 * work lists static valence tails of 0.11 (5s), 12.6 (5p1/2) and 13.4 (5p3/2)
 * that are NOT included (they are strongly frequency dependent for 5p). */
export const CORE_POLARIZABILITY_AU:Record<'ground'|'D1'|'D2',number>={ground:9.1-0.26,D1:9.1,D2:9.1};
/** The Rb I ionization energy above 5S, in cm⁻¹ (NIST ASD; Sansonetti, J. Phys. Chem. Ref. Data 35,
 * 301 (2006)): a photon above it less the 5P energy ionizes 5P, which the model does not hold
 * (5P₃/₂ below 479.1 nm, 5P₁/₂ below 473.7 nm). */
export const IONIZATION_ENERGY_CM=33690.8048;
export const SPECTROSCOPY_SOURCES = ['https://steck.us/alkalidata/rubidium87numbers.pdf','https://arxiv.org/abs/1209.0613','https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator/blob/master/arc/data/rb_NIST_level_data.ascii'];
