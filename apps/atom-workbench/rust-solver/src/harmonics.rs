//! Frequency coordinates and verification utilities for Fourier Lindblad work.
//!
//! A discovered floating-point integer relation is not proof of physical phase
//! locking. Exact round trips and ULP-compatible relations are distinguished;
//! the latter must remain visible in finite-time error budgets and infinity
//! claims. No detuning-, rate-, or amplitude-based physical cutoff is used.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct FrequencyBasis {
    pub frequencies: Vec<f64>,
    /// Original input order; omega_j ~= sum_k index[j][k] * frequencies[k].
    pub drive_indices: Vec<Vec<i32>>,
    pub reconstructed_frequencies: Vec<f64>,
    /// Compensated reconstructed frequency minus redundant/input omega,
    /// including sub-ULP terms and declared-unit conversion roundoff.
    pub reconstruction_errors: Vec<f64>,
    pub exact_relations: usize,
    pub rounded_relations: usize,
    /// For inferred numeric input: every compensated frequency residual is
    /// exactly zero. For declared carrier coordinates: phase relations are
    /// exact by definition; redundant omega literals may have rounding error.
    pub exact: bool,
    /// Low-order combination search stopped at its bounded work budget.
    pub search_exhausted: bool,
    /// source_to_basis[source carrier axis][final Fourier basis axis]. Only
    /// exact integer source-frequency reductions may change this mapping.
    pub source_to_basis:Option<Vec<Vec<i32>>>,
}
impl FrequencyBasis{
    pub fn source(&self)->&'static str{if self.source_to_basis.is_some(){"declared-carrier-coordinates"}else{"inferred-numeric"}}
}

fn half_ulp_up(x: f64) -> f64 {
    let gap = (x.next_up() - x).abs();
    if gap.is_finite() {
        gap * 0.5
    } else {
        (x - x.next_down()).abs() * 0.5
    }
}
fn half_ulp_down(x: f64) -> f64 {
    (x - x.next_down()).abs() * 0.5
}
#[cfg(test)]
fn half_ulp(x: f64) -> f64 {
    half_ulp_up(x).max(half_ulp_down(x))
}

/// Compensated dot product, retaining product roundoff through fma.
fn dot_parts(index: &[i32], basis: &[f64]) -> (f64, f64) {
    let mut hi = 0.;
    let mut lo = 0.;
    for (&n, &f) in index.iter().zip(basis) {
        let a = n as f64;
        let product = a * f;
        let product_error = a.mul_add(f, -product);
        let sum = hi + product;
        let z = sum - hi;
        let sum_error = (hi - (sum - z)) + (product - z);
        hi = sum;
        lo += product_error + sum_error;
    }
    (hi, lo)
}
fn reconstruction(index: &[i32], basis: &[f64]) -> f64 {
    let (hi, lo) = dot_parts(index, basis);
    hi + lo
}
fn relation(input: f64, index: &[i32], basis: &[f64], allow_rounded: bool) -> bool {
    let (hi, lo) = dot_parts(index, basis);
    // A rounded f64 round-trip is insufficient: hi can equal input while a
    // nonzero low part contains a real beat. That beat matters at t->infinity.
    if (input - hi) - lo == 0. {
        return true;
    }
    if !allow_rounded {
        return false;
    }
    // Strict interval overlap: touching at a rounding boundary is insufficient.
    let residual = (input - hi) - lo;
    // IEEE spacing is asymmetric at powers of two. Using max(up,down) would
    // wrongly consume a genuine adjacent-float beat on the denser side.
    let uncertainty = if residual > 0. {
        half_ulp_down(input)
    } else {
        half_ulp_up(input)
    } + index
        .iter()
        .zip(basis)
        .map(|(&n, &f)| {
            let upward = (residual > 0.) == (n >= 0);
            (n as f64).abs()
                * if upward {
                    half_ulp_up(f)
                } else {
                    half_ulp_down(f)
                }
        })
        .sum::<f64>();
    residual.abs() < uncertainty
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let c = a % b;
        a = b;
        b = c;
    }
    a.abs()
}

fn search_combination(
    input: f64,
    basis: &[f64],
    bound: i32,
    rounded: bool,
    exhausted: &mut bool,
) -> Option<Vec<i32>> {
    // Single-axis multiples are cheap even when the requested relation order is
    // high; multi-axis enumeration is by increasing L1 cost, with bounded work.
    for j in 0..basis.len() {
        for n in 1..=bound {
            let mut index = vec![0; basis.len()];
            index[j] = n;
            if relation(input, &index, basis, rounded) {
                return Some(index);
            }
        }
    }
    fn visit(
        axis: usize,
        left: i32,
        index: &mut [i32],
        input: f64,
        basis: &[f64],
        rounded: bool,
        visits: &mut usize,
    ) -> Option<Vec<i32>> {
        if *visits >= 200_000 {
            return None;
        }
        if axis == index.len() {
            if left != 0 || index.iter().filter(|&&n| n != 0).count() < 2 {
                return None;
            }
            *visits += 1;
            return relation(input, index, basis, rounded).then(|| index.to_vec());
        }
        for n in -left..=left {
            index[axis] = n;
            if let Some(found) = visit(
                axis + 1,
                left - n.abs(),
                index,
                input,
                basis,
                rounded,
                visits,
            ) {
                return Some(found);
            }
            if *visits >= 200_000 {
                break;
            }
        }
        index[axis] = 0;
        None
    }
    let mut visits = 0;
    for cost in 2..=bound {
        if let Some(found) = visit(
            0,
            cost,
            &mut vec![0; basis.len()],
            input,
            basis,
            rounded,
            &mut visits,
        ) {
            return Some(found);
        }
        if visits >= 200_000 {
            *exhausted = true;
            break;
        }
    }
    None
}

/// Discover low-order integer relationships without a physical frequency cutoff.
/// Pair p:q relations may refine a fundamental; small integer sums/differences
/// may reduce additional axes. Undiscovered relationships retain their original
/// independent coordinates. All input frequencies must be positive and finite.
pub fn frequency_basis(input: &[f64], max_integer: i32) -> Result<FrequencyBasis, String> {
    if !(1..=64).contains(&max_integer) {
        return Err("Integer relationship search bound must be in [1,64]".into());
    }
    if input.iter().any(|&x| !x.is_finite() || x <= 0.) {
        return Err("Drive frequencies must be positive and finite".into());
    }
    let mut sequence: Vec<usize> = (0..input.len()).collect();
    sequence.sort_by(|&a, &b| input[a].total_cmp(&input[b]));
    let mut basis = Vec::<f64>::new();
    let mut mapping: Vec<Vec<i32>> = vec![vec![]; input.len()];
    let mut processed = Vec::<usize>::new();
    let mut exact_relations = 0;
    let mut rounded_relations = 0;
    let mut exhausted = false;
    for current in sequence {
        let omega = input[current];
        let mut found = None;
        // Always exhaust the bounded exact search before accepting roundoff.
        for rounded in [false, true] {
            if basis.is_empty() {
                break;
            }
            if let Some(index) =
                search_combination(omega, &basis, max_integer, rounded, &mut exhausted)
            {
                found = Some(index);
                if rounded {
                    rounded_relations += 1;
                } else {
                    exact_relations += 1;
                }
                break;
            }
            'pairs: for axis in 0..basis.len() {
                for denominator in 2..=max_integer {
                    for numerator in 1..=max_integer {
                        if gcd(numerator, denominator) != 1 {
                            continue;
                        }
                        let mut candidate_basis = basis.clone();
                        candidate_basis[axis] /= denominator as f64;
                        if candidate_basis[axis] == 0. || !candidate_basis[axis].is_finite() {
                            continue;
                        }
                        let mut proposed = vec![0; basis.len()];
                        proposed[axis] = numerator;
                        if !relation(omega, &proposed, &candidate_basis, rounded) {
                            continue;
                        }
                        let mut candidate_mapping = mapping.clone();
                        let mut valid = true;
                        for &previous in &processed {
                            if let Some(value) = mapping[previous][axis].checked_mul(denominator) {
                                candidate_mapping[previous][axis] = value;
                            } else {
                                valid = false;
                                break;
                            }
                            if !relation(
                                input[previous],
                                &candidate_mapping[previous],
                                &candidate_basis,
                                rounded,
                            ) {
                                valid = false;
                                break;
                            }
                        }
                        if !valid {
                            continue;
                        }
                        // For an ULP pair, require a nonempty common interval of a
                        // single candidate-fundamental correction. Other axes remain
                        // fixed; this is more conservative than independent tolerances.
                        if rounded {
                            let f = candidate_basis[axis];
                            let mut lower = -half_ulp_down(f);
                            let mut upper = half_ulp_up(f);
                            for (&original, index) in processed
                                .iter()
                                .map(|&p| (&input[p], &candidate_mapping[p]))
                                .chain(std::iter::once((&omega, &proposed)))
                            {
                                let multiplier = index[axis] as f64;
                                if multiplier == 0. {
                                    continue;
                                }
                                let (hi, lo) = dot_parts(index, &candidate_basis);
                                let residual = (original - hi) - lo;
                                let a = (residual - half_ulp_down(original)) / multiplier;
                                let b = (residual + half_ulp_up(original)) / multiplier;
                                lower = lower.max(a.min(b));
                                upper = upper.min(a.max(b));
                            }
                            if !(lower < upper) {
                                continue;
                            }
                            candidate_basis[axis] += lower + (upper - lower) / 2.;
                        }
                        basis = candidate_basis;
                        mapping = candidate_mapping;
                        found = Some(proposed);
                        if rounded {
                            rounded_relations += 1;
                        } else {
                            exact_relations += 1;
                        }
                        break 'pairs;
                    }
                }
            }
            if found.is_some() {
                break;
            }
        }
        if let Some(index) = found {
            mapping[current] = index;
        } else {
            basis.push(omega);
            for &previous in &processed {
                mapping[previous].push(0);
            }
            let mut index = vec![0; basis.len()];
            index[basis.len() - 1] = 1;
            mapping[current] = index;
        }
        processed.push(current);
    }
    let reconstructed: Vec<_> = mapping
        .iter()
        .map(|index| reconstruction(index, &basis))
        .collect();
    if reconstructed.iter().any(|x| !x.is_finite()) {
        return Err("Frequency reconstruction overflowed".into());
    }
    let errors: Vec<_> = mapping
        .iter()
        .zip(input)
        .map(|(index, &original)| {
            let (hi, lo) = dot_parts(index, &basis);
            (hi - original) + lo
        })
        .collect();
    let exact = errors.iter().all(|&e| e == 0.);
    if !exact {
        rounded_relations = rounded_relations.max(1);
    }
    Ok(FrequencyBasis {
        frequencies: basis,
        drive_indices: mapping,
        reconstructed_frequencies: reconstructed,
        reconstruction_errors: errors,
        exact_relations,
        rounded_relations,
        exact,
        search_exhausted: exhausted,
        source_to_basis:None,
    })
}

const MAX_DECLARED_INDEX:i32=1_000_000;
fn identity_indices(n:usize)->Vec<Vec<i32>>{(0..n).map(|i|(0..n).map(|j|i32::from(i==j)).collect()).collect()}
fn declared_rows(rows:&[Vec<i32>],count:usize,axes:usize,name:&str)->Result<(),String>{
    if rows.len()!=count||rows.iter().any(|row|row.len()!=axes||row.iter().any(|&value|value.unsigned_abs()>MAX_DECLARED_INDEX as u32)){
        return Err(format!("frequencyCoordinates.{name} has invalid dimensions or integer range"));
    }Ok(())
}
fn compose_indices(rows:&[Vec<i32>],mapping:&[Vec<i32>],width:usize)->Result<Vec<Vec<i32>>,String>{
    rows.iter().map(|row|(0..width).map(|column|{
        let value=row.iter().zip(mapping).try_fold(0i64,|sum,(&n,map)|{
            (n as i64).checked_mul(map[column]as i64).and_then(|product|sum.checked_add(product)).ok_or("Declared frequency index composition overflow")
        })?;
        if value.unsigned_abs()>MAX_DECLARED_INDEX as u64{return Err("Reduced frequency coordinate exceeds the supported integer range".into());}
        Ok(value as i32)
    }).collect()).collect()
}
fn scaled_parts((hi,lo):(f64,f64),scale:f64)->(f64,f64){
    let product=hi*scale;(product,hi.mul_add(scale,-product)+lo*scale)
}
fn ulp_spacing(value:f64)->f64{2.*half_ulp_up(value).max(half_ulp_down(value))}
/// Carrier coordinates are authoritative PHASE provenance, not a tolerance
/// for discovering relationships among independently entered drive omegas.
/// A redundant omega is checked only for floating arithmetic consistency.
pub fn model_frequency_basis(model:&crate::Model,max_integer:i32)->Result<FrequencyBasis,String>{
    let Some(declared)=&model.frequency_coordinates else{
        return frequency_basis(&model.drives.iter().map(|drive|drive.omega).collect::<Vec<_>>(),max_integer);
    };
    let axes=declared.frequencies.len();
    if axes>64||declared.frequencies.iter().any(|value|!value.is_finite()||*value<=0.){
        return Err("frequencyCoordinates.frequencies must contain at most 64 positive finite source axes".into());
    }
    let(source_frequencies,unit)=if let Some(mhz)=&declared.frequencies_mhz{
        if mhz.len()!=axes||mhz.iter().any(|value|!value.is_finite()||*value<=0.){
            return Err("frequencyCoordinates.frequenciesMHz must match the positive finite source axes".into());
        }
        for(&frequency,&angular)in mhz.iter().zip(&declared.frequencies){
            let(hi,lo)=scaled_parts((frequency,0.),std::f64::consts::TAU);
            let error=(hi-angular)+lo;let budget=4.*(ulp_spacing(hi)+ulp_spacing(angular));
            if !error.is_finite()||!budget.is_finite()||error.abs()>budget{return Err("Declared angular frequencies disagree with authoritative frequenciesMHz".into());}
        }
        (mhz.as_slice(),std::f64::consts::TAU)
    }else{(declared.frequencies.as_slice(),1.)};
    declared_rows(&declared.drive_indices,model.drives.len(),axes,"driveIndices")?;
    declared_rows(&declared.frame_indices,model.dimension,axes,"frameIndices")?;
    if declared.frame_sectors.len()!=model.dimension||declared.frame_sectors.iter().any(|&sector|sector!=0&&sector!=1){
        return Err("frequencyCoordinates.frameSectors must contain one ground/excited sector (0/1) per state".into());
    }
    for(drive,index)in model.drives.iter().zip(&declared.drive_indices){
        if !drive.omega.is_finite()||drive.omega<=0.||index.iter().all(|&i|i==0){return Err("Declared drive frequency/index must be positive, finite and nonzero".into());}
        let(hi,lo)=scaled_parts(dot_parts(index,source_frequencies),unit);let value=hi+lo;
        if !value.is_finite()||value<=0.{return Err("Declared drive coordinates do not define a positive finite frequency".into());}
        let error=(hi-drive.omega)+lo;
        // Four ULPs of the actual sum operands/output, including cancellation.
        // This validates a redundant numeric dot; it NEVER merges source axes.
        let budget=4.*(ulp_spacing(drive.omega)+ulp_spacing(value)+index.iter().zip(source_frequencies)
            .filter(|(&n,_)|n!=0).map(|(&n,&f)|ulp_spacing(n as f64*f*unit)).sum::<f64>());
        if !error.is_finite()||!budget.is_finite()||error.abs()>budget{return Err(format!("Drive omega disagrees with its declared frequencyCoordinates dot product ({error:e}, rounding budget {budget:e})"));}
    }
    let reduction=frequency_basis(source_frequencies,max_integer)?;
    // The old numeric finder reports ULP-only relationships separately. Such
    // relationships cannot change the declared source phases: retain those
    // source axes independently, in their original order.
    let(source_basis,mapping,exact_relations)=if reduction.exact&&reduction.rounded_relations==0{
        (reduction.frequencies,reduction.drive_indices,reduction.exact_relations)
    }else{(source_frequencies.to_vec(),identity_indices(axes),0)};
    let frequencies:Vec<_>=source_basis.iter().map(|value|value*unit).collect();
    if frequencies.iter().any(|value|!value.is_finite()||*value<=0.){return Err("Declared source unit conversion overflowed".into());}
    let drive_indices=compose_indices(&declared.drive_indices,&mapping,frequencies.len())?;
    // Validate frame composition too, including static/no-drive models.
    compose_indices(&declared.frame_indices,&mapping,frequencies.len())?;
    let reconstructed_frequencies:Vec<_>=drive_indices.iter().map(|row|{let(hi,lo)=scaled_parts(dot_parts(row,&source_basis),unit);hi+lo}).collect();
    let reconstruction_errors=drive_indices.iter().zip(&model.drives).map(|(row,drive)|{
        let(hi,lo)=scaled_parts(dot_parts(row,&source_basis),unit);(hi-drive.omega)+lo
    }).collect();
    Ok(FrequencyBasis{frequencies,drive_indices,reconstructed_frequencies,reconstruction_errors,
        exact_relations,rounded_relations:0,exact:true,search_exhausted:reduction.search_exhausted,source_to_basis:Some(mapping)})
}
/// Phase rows in the ORIGINAL declared axes, with their authoritative unit
/// conversion. State gauges may contain zero or negative integer entries.
pub fn model_phase_frequency_parts(model:&crate::Model,rows:&[Vec<i32>])->Result<Vec<(f64,f64)>,String>{
    model_frequency_basis(model,16)?;
    let legacy:Vec<_>=model.drives.iter().map(|drive|drive.omega).collect();
    let(source,unit)=if let Some(declared)=&model.frequency_coordinates{
        declared.frequencies_mhz.as_ref().map(|values|(values.as_slice(),std::f64::consts::TAU))
            .unwrap_or((declared.frequencies.as_slice(),1.))
    }else{(legacy.as_slice(),1.)};
    if source.iter().any(|value|!value.is_finite()||*value<=0.)||rows.iter().any(|row|row.len()!=source.len()){
        return Err("Numerical phase rows do not match the finite original source axes".into());
    }
    rows.iter().map(|row|{
        let parts=scaled_parts(dot_parts(row,source),unit);
        if !parts.0.is_finite()||!parts.1.is_finite(){Err("Numerical phase frequency overflow".into())}else{Ok(parts)}
    }).collect()
}

#[cfg(test)]
mod declared_coordinate_tests{
    use super::*;
    use crate::{Drive,FrequencyCoordinates,Model,Operator};
    fn model(frequencies:Vec<f64>,indices:Vec<Vec<i32>>,omegas:Vec<f64>)->Model{
        let axes=frequencies.len();let zero=Operator{real:vec![0.;4],imag:vec![0.;4]};
        Model{dimension:2,states:vec![],h_real:vec![0.;4],h_imag:vec![0.;4],collapse:vec![],
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:!omegas.is_empty(),
            drives:omegas.into_iter().map(|omega|Drive{omega,cos:zero.clone(),sin:zero.clone()}).collect(),
            frequency_coordinates:Some(FrequencyCoordinates{frequencies,frequencies_mhz:None,drive_indices:indices,
                frame_indices:vec![vec![0;axes];2],frame_sectors:vec![0,1]})}
    }
    #[test]
    fn declared_triangle_uses_two_source_axes_despite_redundant_omega_rounding(){
        let model=model(vec![1675.4239285300514,41268.1534315396],vec![vec![0,1],vec![1,1],vec![1,0]],
            vec![41268.1534315396,42943.57736006965,1675.4239285300514]);
        let basis=model_frequency_basis(&model,16).unwrap();
        assert_eq!(basis.frequencies.len(),2);assert_eq!(basis.drive_indices,[vec![0,1],vec![1,1],vec![1,0]]);
        assert_eq!(basis.source_to_basis,Some(vec![vec![1,0],vec![0,1]]));
        assert_eq!(basis.source(),"declared-carrier-coordinates");assert!(basis.exact);assert_eq!(basis.rounded_relations,0);
        assert_ne!(basis.reconstruction_errors[1],0.); // declared phase identity, not a zero literal residual
        let rows=model.frequency_coordinates.as_ref().unwrap().drive_indices.clone();let parts=model_phase_frequency_parts(&model,&rows).unwrap();assert_ne!(parts[1].1,0.);
        let mut legacy=model;legacy.frequency_coordinates=None;
        let old=model_frequency_basis(&legacy,16).unwrap();assert!(old.source_to_basis.is_none());assert_eq!(old.source(),"inferred-numeric");
        assert!(!old.exact||old.frequencies.len()==3);
    }
    #[test]
    fn declared_close_and_ulp_only_source_relationships_remain_independent(){
        for frequencies in [vec![1.,1_f64.next_up()],vec![0.1,0.3]]{
            let model=model(frequencies.clone(),vec![vec![1,0],vec![0,1]],frequencies.clone());
            let basis=model_frequency_basis(&model,16).unwrap();
            assert_eq!(basis.frequencies,frequencies);assert_eq!(basis.source_to_basis,Some(vec![vec![1,0],vec![0,1]]));
            assert!(basis.exact);assert_eq!(basis.rounded_relations,0);
        }
    }
    #[test]
    fn declared_exact_double_frequency_and_static_frame_mapping_are_preserved(){
        let mut driven=model(vec![3.,6.],vec![vec![1,0],vec![0,1],vec![1,1]],vec![3.,6.,9.]);
        driven.frequency_coordinates.as_mut().unwrap().frame_indices=vec![vec![1,-1],vec![0,0]];
        let basis=model_frequency_basis(&driven,16).unwrap();
        assert_eq!(basis.frequencies,[3.]);assert_eq!(basis.source_to_basis,Some(vec![vec![1],vec![2]]));
        assert_eq!(basis.drive_indices,[vec![1],vec![2],vec![3]]);
        driven.drives.clear();driven.frequency_coordinates.as_mut().unwrap().drive_indices.clear();
        let static_basis=model_frequency_basis(&driven,16).unwrap();
        assert_eq!(static_basis.source_to_basis,basis.source_to_basis);assert!(static_basis.drive_indices.is_empty());
        assert_eq!(compose_indices(&driven.frequency_coordinates.as_ref().unwrap().frame_indices,static_basis.source_to_basis.as_ref().unwrap(),1).unwrap(),[vec![-1],vec![0]]);
        let empty=model(vec![],vec![],vec![]);let basis=model_frequency_basis(&empty,16).unwrap();assert_eq!(basis.source_to_basis,Some(vec![]));
    }
    #[test]
    fn declared_mhz_integer_relations_survive_tau_unit_conversion(){
        let tau=std::f64::consts::TAU;
        let mut model=model(vec![100.*tau,300.*tau],vec![vec![1,0],vec![1,1],vec![0,1]],vec![100.*tau,400.*tau,300.*tau]);
        model.frequency_coordinates.as_mut().unwrap().frequencies_mhz=Some(vec![100.,300.]);
        let basis=model_frequency_basis(&model,16).unwrap();
        assert_eq!(basis.frequencies,[100.*tau]);assert_eq!(basis.source_to_basis,Some(vec![vec![1],vec![3]]));
        assert_eq!(basis.drive_indices,[vec![1],vec![4],vec![3]]);assert!(basis.exact);
        let rows=model.frequency_coordinates.as_ref().unwrap().drive_indices.clone();let parts=model_phase_frequency_parts(&model,&rows).unwrap();
        assert_eq!(parts[1],scaled_parts((400.,0.),tau));assert_ne!(parts[1].1,0.);
        // The optional authority changes the unit in which exactness is
        // established; it does not relax the old angular-only numeric test.
        model.frequency_coordinates.as_mut().unwrap().frequencies_mhz=None;
        let angular_only=model_frequency_basis(&model,16).unwrap();assert_eq!(angular_only.frequencies.len(),2);
        model.frequency_coordinates.as_mut().unwrap().frequencies_mhz=Some(vec![100.,301.]);
        assert!(model_frequency_basis(&model,16).is_err());
    }
    #[test]
    fn malformed_declared_metadata_is_rejected_without_guessing_a_replacement(){
        let valid=model(vec![2.],vec![vec![1]],vec![2.]);
        let mut cases=vec![];
        let mut wrong=valid.clone();wrong.drives[0].omega+=1e-5;cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().drive_indices.clear();cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().drive_indices[0]=vec![0];cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().drive_indices[0]=vec![1,0];cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().frame_indices[0]=vec![1_000_001];cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().frame_indices.pop();cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().frame_sectors[0]=2;cases.push(wrong);
        let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().frequencies_mhz=Some(vec![]);cases.push(wrong);
        for invalid in [0.,f64::NAN,f64::INFINITY]{let mut wrong=valid.clone();wrong.frequency_coordinates.as_mut().unwrap().frequencies[0]=invalid;cases.push(wrong);}
        for wrong in cases{assert!(model_frequency_basis(&wrong,16).is_err());}
        for invalid in [serde_json::json!(0.5),serde_json::json!(2_147_483_648_i64)]{
            assert!(serde_json::from_value::<FrequencyCoordinates>(serde_json::json!({"frequencies":[2.],"driveIndices":[[invalid]],"frameIndices":[[0],[0]],"frameSectors":[0,1]})).is_err());
        }
    }
}

/// One coefficient for each conjugate pair, first nonzero index positive.
/// Real physical coordinates follow x(theta)=x0+2 Re sum_n z_n exp(i n.theta).
#[derive(Clone, Debug)]
pub struct FourierCoefficient {
    pub index: Vec<i32>,
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct UniformBound {
    pub per_coordinate: Vec<f64>,
    pub max_coordinate: f64,
    pub l2: f64,
}
pub type RealSparse = Vec<Vec<(usize, f64)>>;
#[derive(Clone, Debug)]
pub struct FourierDrive {
    pub index: Vec<i32>,
    pub cos: RealSparse,
    pub sin: RealSparse,
}
#[derive(Clone, Debug)]
pub struct BoundaryForcing {
    pub coefficients: Vec<FourierCoefficient>,
    pub bound: UniformBound,
}

fn sign(index: &[i32]) -> i32 {
    index
        .iter()
        .find(|&&n| n != 0)
        .map(|&n| if n > 0 { 1 } else { -1 })
        .unwrap_or(1)
}
fn canonical(index: &[i32]) -> Result<Vec<i32>, String> {
    let s = sign(index);
    index
        .iter()
        .map(|&n| {
            n.checked_mul(s)
                .ok_or_else(|| "Harmonic index overflow".into())
        })
        .collect()
}
fn coefficient_map<'a>(
    input: &'a [FourierCoefficient],
    dimensions: usize,
    size: usize,
) -> Result<BTreeMap<Vec<i32>, &'a FourierCoefficient>, String> {
    let mut result = BTreeMap::new();
    for value in input {
        if value.index.len() != dimensions || value.real.len() != size || value.imag.len() != size {
            return Err("Inconsistent Fourier coefficient dimensions".into());
        }
        if sign(&value.index) < 0 {
            return Err("Fourier coefficients must use canonical +/- index representatives".into());
        }
        if value.real.iter().chain(&value.imag).any(|x| !x.is_finite()) {
            return Err("Non-finite Fourier coefficient".into());
        }
        if value.index.iter().all(|&n| n == 0) && value.imag.iter().any(|&x| x != 0.) {
            return Err("The central Hermitian-coordinate coefficient must be real".into());
        }
        if result.insert(value.index.clone(), value).is_some() {
            return Err("Duplicate canonical Fourier coefficient".into());
        }
    }
    Ok(result)
}

/// Uniform, all-phase triangle bound, with absent harmonics interpreted as zero.
/// This is a bound on coordinate differences, not a sampled-grid estimate.
pub fn uniform_difference(
    a: &[FourierCoefficient],
    b: &[FourierCoefficient],
) -> Result<UniformBound, String> {
    let exemplar = a.first().or_else(|| b.first());
    let Some(exemplar) = exemplar else {
        return Ok(UniformBound {
            per_coordinate: vec![],
            max_coordinate: 0.,
            l2: 0.,
        });
    };
    let size = exemplar.real.len();
    let dimensions = exemplar.index.len();
    let left = coefficient_map(a, dimensions, size)?;
    let right = coefficient_map(b, dimensions, size)?;
    let indices: BTreeSet<_> = left.keys().chain(right.keys()).cloned().collect();
    let mut bound = vec![0.; size];
    for index in indices {
        let central = index.iter().all(|&n| n == 0);
        for i in 0..size {
            let ar = left.get(&index).map_or(0., |v| v.real[i]);
            let ai = left.get(&index).map_or(0., |v| v.imag[i]);
            let br = right.get(&index).map_or(0., |v| v.real[i]);
            let bi = right.get(&index).map_or(0., |v| v.imag[i]);
            bound[i] += if central {
                (ar - br).abs()
            } else {
                2. * (ar - br).hypot(ai - bi)
            };
        }
    }
    if bound.iter().any(|x| !x.is_finite()) {
        return Err("Uniform Fourier bound overflowed".into());
    }
    let l2 = bound.iter().fold(0_f64, |norm, &value| norm.hypot(value));
    if !l2.is_finite() {
        return Err("Uniform Fourier norm overflowed".into());
    }
    Ok(UniformBound {
        max_coordinate: bound.iter().copied().fold(0., f64::max),
        l2,
        per_coordinate: bound,
    })
}

fn check_operator(operator: &RealSparse, size: usize) -> Result<(), String> {
    if operator.len() != size
        || operator
            .iter()
            .any(|col| col.iter().any(|&(row, v)| row >= size || !v.is_finite()))
    {
        return Err("Invalid sparse Fourier generator coefficient".into());
    }
    Ok(())
}

/// Exact omitted Fourier forcing of the original differential equation.
/// Retained indices may be canonical or signed; conjugate counterparts are
/// implicit. Sum all paths to each omitted index before taking a norm, so exact
/// destructive interference remains exact. Constant-generator terms cannot
/// create an outside harmonic and are intentionally absent from this routine.
pub fn boundary_forcing(
    drives: &[FourierDrive],
    coefficients: &[FourierCoefficient],
    retained_indices: &[Vec<i32>],
) -> Result<BoundaryForcing, String> {
    let exemplar = coefficients
        .first()
        .ok_or("Boundary forcing requires coefficient dimensions")?;
    let dimensions = exemplar.index.len();
    let size = exemplar.real.len();
    coefficient_map(coefficients, dimensions, size)?;
    let mut retained = BTreeSet::new();
    for index in retained_indices {
        if index.len() != dimensions {
            return Err("Inconsistent retained Fourier index dimensions".into());
        }
        retained.insert(canonical(index)?);
    }
    for drive in drives {
        if drive.index.len() != dimensions || drive.index.iter().all(|&n| n == 0) {
            return Err("Invalid nonzero Fourier drive index".into());
        }
        check_operator(&drive.cos, size)?;
        check_operator(&drive.sin, size)?;
    }
    let mut outside: BTreeMap<Vec<i32>, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for coefficient in coefficients {
        if !retained.contains(&coefficient.index) {
            return Err("Coefficient lies outside the retained Fourier lattice".into());
        }
        let central = coefficient.index.iter().all(|&n| n == 0);
        for source_sign in if central { vec![1] } else { vec![1, -1] } {
            let source: Vec<_> = coefficient.index.iter().map(|&n| n * source_sign).collect();
            for drive in drives {
                for step in [1, -1] {
                    let target: Vec<i32> = source
                        .iter()
                        .zip(&drive.index)
                        .map(|(&n, &shift)| {
                            shift
                                .checked_mul(step)
                                .and_then(|s| n.checked_add(s))
                                .ok_or_else(|| "Boundary Fourier index overflow".to_string())
                        })
                        .collect::<Result<_, _>>()?;
                    if retained.contains(&canonical(&target)?) {
                        continue;
                    }
                    let (real, imag) = outside
                        .entry(target)
                        .or_insert_with(|| (vec![0.; size], vec![0.; size]));
                    // exp(+i n theta): L_+=(Ac-i As)/2, L_-=(Ac+i As)/2.
                    for (col, entries) in drive.cos.iter().enumerate() {
                        for &(row, v) in entries {
                            real[row] += 0.5 * v * coefficient.real[col];
                            imag[row] += 0.5 * v * coefficient.imag[col] * source_sign as f64;
                        }
                    }
                    for (col, entries) in drive.sin.iter().enumerate() {
                        for &(row, v) in entries {
                            real[row] +=
                                0.5 * step as f64 * v * coefficient.imag[col] * source_sign as f64;
                            imag[row] -= 0.5 * step as f64 * v * coefficient.real[col];
                        }
                    }
                }
            }
        }
    }
    let coefficients: Vec<_> = outside
        .into_iter()
        .filter(|(index, _)| sign(index) > 0)
        .map(|(index, (real, imag))| FourierCoefficient { index, real, imag })
        .collect();
    let bound = uniform_difference(&coefficients, &[])?;
    Ok(BoundaryForcing {
        coefficients,
        bound,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn coefficient(index: &[i32], re: &[f64], im: &[f64]) -> FourierCoefficient {
        FourierCoefficient {
            index: index.to_vec(),
            real: re.to_vec(),
            imag: im.to_vec(),
        }
    }
    #[test]
    fn exact_integer_and_rational_relations_reduce_axes() {
        let a = frequency_basis(&[4.6, 2.3, 6.9], 16).unwrap();
        assert_eq!(a.frequencies.len(), 1);
        assert!(a.exact || a.rounded_relations > 0);
        for (actual, expected) in a.reconstructed_frequencies.iter().zip([4.6, 2.3, 6.9]) {
            assert!((actual - expected).abs() <= 2. * half_ulp(expected));
        }
        let rational = frequency_basis(&[6., 10., 15.], 16).unwrap();
        assert_eq!(rational.frequencies, vec![1.]);
        assert_eq!(rational.drive_indices, vec![vec![6], vec![10], vec![15]]);
        assert!(rational.exact);
    }
    #[test]
    fn integer_sum_retains_two_genuine_independent_frequencies() {
        // Adding 0.5 leaves sqrt(2)'s exponent bin unchanged: this sum is an
        // exact relation between the actual input binary floating numbers.
        let a = frequency_basis(&[0.5, 2_f64.sqrt(), 0.5 + 2_f64.sqrt()], 16).unwrap();
        assert_eq!(a.frequencies.len(), 2);
        assert_eq!(a.drive_indices[2], vec![1, 1]);
        assert!(a.exact);
    }
    #[test]
    fn rounded_frequency_roundtrip_retains_its_real_sub_ulp_beat() {
        let original = 3. * 0.1_f64;
        let (hi, lo) = dot_parts(&[3], &[0.1]);
        assert_eq!(hi + lo, original);
        assert_ne!(lo, 0.);
        assert!(!relation(original, &[3], &[0.1], false));
        let basis = frequency_basis(&[0.1, original], 16).unwrap();
        assert!(!basis.exact);
        assert!(basis.rounded_relations > 0);
        assert!(basis.reconstruction_errors.iter().any(|&value| value != 0.));
        let sum = frequency_basis(&[1., 2_f64.sqrt(), 1. + 2_f64.sqrt()], 16).unwrap();
        assert!(!sum.exact);
        assert!(sum.rounded_relations > 0);
    }
    #[test]
    fn small_physical_beat_and_adjacent_ulp_are_not_rounded_away() {
        let omega = 2. * std::f64::consts::PI * 31_530_839.13950622;
        let a = frequency_basis(
            &[omega, 2. * omega + 2. * std::f64::consts::PI * 0.125e-6],
            16,
        )
        .unwrap();
        assert_eq!(a.frequencies.len(), 2);
        assert!(a.exact);
        let a = frequency_basis(&[1., 2_f64.next_up()], 16).unwrap();
        assert_eq!(a.frequencies.len(), 2);
        let a = frequency_basis(&[1., 2_f64.next_down()], 16).unwrap();
        assert_eq!(a.frequencies.len(), 2);
    }
    #[test]
    fn uniform_bound_exposes_aliasing_of_a_one_dimensional_phase_grid() {
        // 1-cos(3 theta1-theta2) vanishes at theta2=3 theta1, but peaks at 2.
        let a = vec![
            coefficient(&[0, 0], &[1.], &[0.]),
            coefficient(&[3, -1], &[-0.5], &[0.]),
        ];
        assert_eq!(uniform_difference(&a, &[]).unwrap().max_coordinate, 2.);
        assert_eq!(uniform_difference(&a, &a).unwrap().max_coordinate, 0.);
    }
    #[test]
    fn uniform_complex_harmonic_bound_is_tight_for_one_mode() {
        let a = vec![coefficient(&[1], &[3.], &[4.])];
        assert_eq!(uniform_difference(&a, &[]).unwrap().max_coordinate, 10.);
    }
    #[test]
    fn boundary_forcing_uses_true_generator_and_preserves_interference() {
        let drive = FourierDrive {
            index: vec![1],
            cos: vec![vec![(0, 2.)]],
            sin: vec![vec![(0, 4.)]],
        };
        let values = vec![
            coefficient(&[0], &[1.], &[0.]),
            coefficient(&[1], &[3.], &[5.]),
        ];
        let report = boundary_forcing(&[drive.clone()], &values, &[vec![0], vec![1]]).unwrap();
        assert_eq!(report.coefficients.len(), 1);
        assert_eq!(report.coefficients[0].index, vec![2]);
        // (1-2i)*(3+5i)=13-i.
        assert_eq!(report.coefficients[0].real, vec![13.]);
        assert_eq!(report.coefficients[0].imag, vec![-1.]);
        assert_eq!(report.bound.max_coordinate, 2. * 13_f64.hypot(1.));
        let opposite = FourierDrive {
            index: vec![1],
            cos: vec![vec![(0, -2.)]],
            sin: vec![vec![(0, -4.)]],
        };
        assert_eq!(
            boundary_forcing(&[drive, opposite], &values, &[vec![0], vec![1]])
                .unwrap()
                .bound
                .max_coordinate,
            0.
        );
    }
    #[test]
    fn malformed_and_nonfinite_inputs_are_rejected() {
        assert!(frequency_basis(&[0.], 16).is_err());
        assert!(frequency_basis(&[f64::NAN], 16).is_err());
        assert!(uniform_difference(&[coefficient(&[-1], &[1.], &[0.])], &[]).is_err());
        assert!(uniform_difference(&[coefficient(&[0], &[1.], &[1.])], &[]).is_err());
    }
}
