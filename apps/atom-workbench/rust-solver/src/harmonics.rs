//! Frequency coordinates of a Model: declared carrier axes, exact integer
//! relations between them, and the phase axes used for Fourier work.
//!
//! A discovered floating-point integer relation is not proof of physical phase
//! locking. Exact round trips and ULP-compatible relations are distinguished;
//! only an exact relation between declared source axes changes the phase
//! basis. No detuning-, rate-, or amplitude-based physical cutoff is used.

struct FrequencyBasis {
    frequencies: Vec<f64>,
    /// Original input order; omega_j ~= sum_k indices[j][k] * frequencies[k].
    indices: Vec<Vec<i32>>,
    exact_relations: usize,
    rounded_relations: usize,
    /// Every compensated frequency residual is exactly zero.
    exact: bool,
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

fn search_combination(input: f64, basis: &[f64], bound: i32, rounded: bool) -> Option<Vec<i32>> {
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
            break;
        }
    }
    None
}

/// Discover low-order integer relationships without a physical frequency cutoff.
/// Pair p:q relations may refine a fundamental; small integer sums/differences
/// may reduce additional axes. Undiscovered relationships retain their original
/// independent coordinates. All input frequencies must be positive and finite.
fn frequency_basis(input: &[f64], max_integer: i32) -> Result<FrequencyBasis, String> {
    if !(1..=64).contains(&max_integer) {
        return Err("Integer relationship search bound must be in [1,64]".into());
    }
    if input.iter().any(|&x| !x.is_finite() || x <= 0.) {
        return Err("Frequencies must be positive and finite".into());
    }
    let mut sequence: Vec<usize> = (0..input.len()).collect();
    sequence.sort_by(|&a, &b| input[a].total_cmp(&input[b]));
    let mut basis = Vec::<f64>::new();
    let mut mapping: Vec<Vec<i32>> = vec![vec![]; input.len()];
    let mut processed = Vec::<usize>::new();
    let mut exact_relations = 0;
    let mut rounded_relations = 0;
    for current in sequence {
        let omega = input[current];
        let mut found = None;
        // Always exhaust the bounded exact search before accepting roundoff.
        for rounded in [false, true] {
            if basis.is_empty() {
                break;
            }
            if let Some(index) = search_combination(omega, &basis, max_integer, rounded) {
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
    if mapping.iter().any(|index| !reconstruction(index, &basis).is_finite()) {
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
        indices: mapping,
        exact_relations,
        rounded_relations,
        exact,
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
/// Phase axes of a declared-coordinate Model for Fourier work. Without an
/// exact integer relation between the declared source axes these ARE the
/// source axes (identity mapping, unchanged order). With one (equally spaced
/// carriers, for example) the exact reduced basis is used, so that a Fourier
/// charge has zero frequency only when it is zero; otherwise the envelope
/// null space would be degenerate and a long-time mean ill-defined.
pub struct PhaseAxes{
    /// Compensated angular frequency (hi, lo) of every axis.
    pub parts:Vec<(f64,f64)>,
    pub drive_indices:Vec<Vec<i32>>,
    /// Laboratory frame index of every state in these axes.
    pub frame_indices:Vec<Vec<i32>>,
    pub frame_sectors:Vec<i32>,
    /// source_to_basis[source axis][phase axis].
    pub source_to_basis:Vec<Vec<i32>>,
}
/// Carrier coordinates are authoritative PHASE provenance, not a tolerance
/// for discovering relationships among independently entered drive omegas.
/// A redundant omega is checked only for floating arithmetic consistency.
pub fn model_phase_axes(model:&crate::Model)->Result<PhaseAxes,String>{
    let declared=model.frequency_coordinates.as_ref().ok_or("Phase axes require declared frequencyCoordinates")?;
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
    let reduction=frequency_basis(source_frequencies,16)?;
    // A ULP-only relationship cannot change the declared source phases, and
    // without any relationship there is nothing to reduce: retain the source
    // axes independently, in their original order.
    let(basis,source_to_basis)=if reduction.exact&&reduction.rounded_relations==0&&reduction.exact_relations>0{
        (reduction.frequencies,reduction.indices)
    }else{(source_frequencies.to_vec(),identity_indices(axes))};
    let parts:Vec<_>=basis.iter().map(|&value|scaled_parts((value,0.),unit)).collect();
    if parts.iter().any(|&(hi,_)|!hi.is_finite()||hi<=0.){return Err("Declared source unit conversion overflowed".into());}
    let drive_indices=compose_indices(&declared.drive_indices,&source_to_basis,parts.len())?;
    // Frame composition is validated too, including static/no-drive models.
    let frame_indices=compose_indices(&declared.frame_indices,&source_to_basis,parts.len())?;
    Ok(PhaseAxes{parts,drive_indices,frame_indices,frame_sectors:declared.frame_sectors.clone(),source_to_basis})
}

#[cfg(test)]
mod declared_coordinate_tests{
    use super::*;
    use crate::{Drive,FrequencyCoordinates,Model,Operator};
    fn model(frequencies:Vec<f64>,indices:Vec<Vec<i32>>,omegas:Vec<f64>)->Model{
        let axes=frequencies.len();let zero=Operator{real:vec![0.;4],imag:vec![0.;4]};
        Model{dimension:2,states:vec![],h_real:vec![0.;4],h_imag:vec![0.;4],collapse:vec![],
            initial_index:0,initial_real:None,initial_imag:None,
            drives:omegas.into_iter().map(|omega|Drive{omega,cos:zero.clone(),sin:zero.clone()}).collect(),
            frequency_coordinates:Some(FrequencyCoordinates{frequencies,frequencies_mhz:None,drive_indices:indices,
                frame_indices:vec![vec![0;axes];2],frame_sectors:vec![0,1]})}
    }
    #[test]
    fn declared_triangle_uses_two_source_axes_despite_redundant_omega_rounding(){
        let omegas=vec![41268.1534315396,42943.57736006965,1675.4239285300514];
        let model=model(vec![1675.4239285300514,41268.1534315396],vec![vec![0,1],vec![1,1],vec![1,0]],omegas.clone());
        let axes=model_phase_axes(&model).unwrap();
        assert_eq!(axes.parts,[(1675.4239285300514,0.),(41268.1534315396,0.)]);assert_eq!(axes.drive_indices,[vec![0,1],vec![1,1],vec![1,0]]);
        assert_eq!(axes.source_to_basis,[vec![1,0],vec![0,1]]);
        // Declared phase identity, not a zero literal residual.
        assert_ne!(dot_parts(&axes.drive_indices[1],&[axes.parts[0].0,axes.parts[1].0]).1,0.);
        // The rounded omega literals alone carry no exact two-axis relation.
        let numeric=frequency_basis(&omegas,16).unwrap();assert!(!numeric.exact||numeric.frequencies.len()==3);
    }
    #[test]
    fn declared_close_and_ulp_only_source_relationships_remain_independent(){
        for frequencies in [vec![1.,1_f64.next_up()],vec![0.1,0.3]]{
            let model=model(frequencies.clone(),vec![vec![1,0],vec![0,1]],frequencies.clone());
            let axes=model_phase_axes(&model).unwrap();
            assert_eq!(axes.parts,frequencies.iter().map(|&f|(f,0.)).collect::<Vec<_>>());assert_eq!(axes.source_to_basis,[vec![1,0],vec![0,1]]);
        }
    }
    #[test]
    fn declared_exact_double_frequency_and_static_frame_mapping_are_preserved(){
        let mut driven=model(vec![3.,6.],vec![vec![1,0],vec![0,1],vec![1,1]],vec![3.,6.,9.]);
        driven.frequency_coordinates.as_mut().unwrap().frame_indices=vec![vec![1,-1],vec![0,0]];
        let axes=model_phase_axes(&driven).unwrap();
        assert_eq!(axes.parts,[(3.,0.)]);assert_eq!(axes.source_to_basis,[vec![1],vec![2]]);
        assert_eq!(axes.drive_indices,[vec![1],vec![2],vec![3]]);assert_eq!(axes.frame_indices,[vec![-1],vec![0]]);
        driven.drives.clear();driven.frequency_coordinates.as_mut().unwrap().drive_indices.clear();
        let static_axes=model_phase_axes(&driven).unwrap();
        assert_eq!(static_axes.source_to_basis,axes.source_to_basis);assert!(static_axes.drive_indices.is_empty());
        assert_eq!(static_axes.frame_indices,axes.frame_indices);
        let empty=model_phase_axes(&model(vec![],vec![],vec![])).unwrap();assert!(empty.parts.is_empty()&&empty.source_to_basis.is_empty());
    }
    #[test]
    fn declared_mhz_integer_relations_survive_tau_unit_conversion(){
        let tau=std::f64::consts::TAU;
        let mut model=model(vec![100.*tau,300.*tau],vec![vec![1,0],vec![1,1],vec![0,1]],vec![100.*tau,400.*tau,300.*tau]);
        model.frequency_coordinates.as_mut().unwrap().frequencies_mhz=Some(vec![100.,300.]);
        let axes=model_phase_axes(&model).unwrap();
        assert_eq!(axes.parts,[scaled_parts((100.,0.),tau)]);assert_ne!(axes.parts[0].1,0.);
        assert_eq!(axes.source_to_basis,[vec![1],vec![3]]);assert_eq!(axes.drive_indices,[vec![1],vec![4],vec![3]]);
        // The optional authority changes the unit in which exactness is
        // established; it does not relax the angular-only numeric test.
        model.frequency_coordinates.as_mut().unwrap().frequencies_mhz=None;
        assert_eq!(model_phase_axes(&model).unwrap().parts.len(),2);
        model.frequency_coordinates.as_mut().unwrap().frequencies_mhz=Some(vec![100.,301.]);
        assert!(model_phase_axes(&model).is_err());
    }
    #[test]
    fn phase_axis_parts_keep_their_recorded_bits(){
        // Compensated parts are pinned exactly, signed zero low parts included.
        let tau=std::f64::consts::TAU;
        let angular=model(vec![100.*tau,300.*tau],vec![vec![1,0],vec![1,1],vec![0,1]],vec![100.*tau,400.*tau,300.*tau]);
        let mut mhz=angular.clone();mhz.frequency_coordinates.as_mut().unwrap().frequencies_mhz=Some(vec![100.,300.]);
        for(fixture,expected)in[
            (model(vec![1675.4239285300514,41268.1534315396],vec![vec![0,1],vec![1,1],vec![1,0]],vec![41268.1534315396,42943.57736006965,1675.4239285300514]),
                vec![(0x409a2db21a5211a6,0),(0x40e42684e8e94298,0)]),
            (model(vec![0.1,0.3],vec![vec![1,0],vec![0,1]],vec![0.1,0.3]),vec![(0x3fb999999999999a,0),(0x3fd3333333333333,0)]),
            (model(vec![3.,6.],vec![vec![1,0],vec![0,1],vec![1,1]],vec![3.,6.,9.]),vec![(0x4008000000000000,0)]),
            (mhz,vec![(0x4083a28c59d5433b,0xbd20000000000000)]),
            (angular,vec![(0x4083a28c59d5433b,0),(0x409d73d286bfe4d8,0)]),
        ]{
            let bits:Vec<(u64,u64)>=model_phase_axes(&fixture).unwrap().parts.iter().map(|&(hi,lo)|(hi.to_bits(),lo.to_bits())).collect();
            assert_eq!(bits,expected);
        }
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
        for wrong in cases{assert!(model_phase_axes(&wrong).is_err());}
        for invalid in [serde_json::json!(0.5),serde_json::json!(2_147_483_648_i64)]{
            assert!(serde_json::from_value::<FrequencyCoordinates>(serde_json::json!({"frequencies":[2.],"driveIndices":[[invalid]],"frameIndices":[[0],[0]],"frameSectors":[0,1]})).is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_integer_and_rational_relations_reduce_axes() {
        let a = frequency_basis(&[4.6, 2.3, 6.9], 16).unwrap();
        assert_eq!(a.frequencies.len(), 1);
        assert!(a.exact || a.rounded_relations > 0);
        for (index, expected) in a.indices.iter().zip([4.6, 2.3, 6.9]) {
            let actual = reconstruction(index, &a.frequencies);
            assert!((actual - expected).abs() <= 2. * half_ulp(expected));
        }
        let rational = frequency_basis(&[6., 10., 15.], 16).unwrap();
        assert_eq!(rational.frequencies, vec![1.]);
        assert_eq!(rational.indices, vec![vec![6], vec![10], vec![15]]);
        assert!(rational.exact);
    }
    #[test]
    fn integer_sum_retains_two_genuine_independent_frequencies() {
        // Adding 0.5 leaves sqrt(2)'s exponent bin unchanged: this sum is an
        // exact relation between the actual input binary floating numbers.
        let a = frequency_basis(&[0.5, 2_f64.sqrt(), 0.5 + 2_f64.sqrt()], 16).unwrap();
        assert_eq!(a.frequencies.len(), 2);
        assert_eq!(a.indices[2], vec![1, 1]);
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
        assert!(basis.indices.iter().zip([0.1, original]).any(|(index, input)| {
            let (hi, lo) = dot_parts(index, &basis.frequencies);
            (hi - input) + lo != 0.
        }));
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
    fn malformed_and_nonfinite_inputs_are_rejected() {
        assert!(frequency_basis(&[0.], 16).is_err());
        assert!(frequency_basis(&[f64::NAN], 16).is_err());
    }
}
