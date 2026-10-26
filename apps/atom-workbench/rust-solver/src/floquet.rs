//! Fourier-Galerkin solution of the full time-dependent Lindblad equation.
//! No electronic state or density coherence is eliminated. Harmonic order is
//! increased and the reconstructed physical density is checked for convergence.
use super::*;
use crate::harmonics::{FourierCoefficient, FourierDrive, FrequencyBasis};
use faer::{
    sparse::{SparseColMat, Triplet},
    Mat,
};
use std::cell::RefCell;
#[derive(Clone,Debug,Eq,PartialEq,Hash)]
struct RefinementCoordinate{harmonic:Vec<i32>,part:usize,physical:usize}
#[derive(Clone,PartialEq)]
struct RefinementLayout{coordinates:Vec<RefinementCoordinate>,weights:Vec<f64>,frequencies:Vec<f64>,drive_indices:Vec<Vec<i32>>,physical_dimension:usize}
struct ReusableStationary{
    layout:RefinementLayout,trace:RefinementCoordinate,matrix:Sparse,
    prepared:std::rc::Rc<crate::block_solve::Prepared>,
}
// Separate from shifted trajectory factors: the stationary trace-row equation
// has a different operator and may never be reused across a numerical shift.
thread_local! {static MEAN_REFINEMENT_PRECONDITIONER:RefCell<Option<ReusableStationary>>=const{RefCell::new(None)};}
pub(super) fn clear_request_results() {
    MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
    STATIONARY_OPERATORS.with(|v| v.borrow_mut().clear());
    STATIONARY_SYMBOLIC.with(|v| v.borrow_mut().clear());
    STATIONARY_NUMERIC.with(|v| *v.borrow_mut() = None);
}
/// Long-time (phase-averaged) mean of the full driven Lindblad equation.
/// Only steadyOnly requests use this solver; finite-time trajectories are
/// computed by the population core or by the complete-period lane.
pub(super) fn solve_full(lap: &Lapack, model: &Model, options: &Options) -> Result<Value, String> {
    if !options.steady_only {
        return Err("The Fourier-Galerkin solver computes only the long-time mean (steadyOnly)".into());
    }
    MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
    STATIONARY_PROFILE.with(|p| *p.borrow_mut() = Some(Vec::new()));
    let result = solve(lap, model, options).map(|mut result| {
        result["diagnostics"]["stationaryPhases"] =
            STATIONARY_PROFILE.with(|p| json!(p.borrow_mut().take()));
        result
    });
    MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
    result
}
use std::collections::{BTreeSet, HashMap};

type Sparse = Vec<Vec<(usize, f64)>>;
thread_local! {static STATIONARY_PROFILE:RefCell<Option<Vec<Value>>>=const{RefCell::new(None)};}
struct StationarySymbolic {
    dimension: usize,
    column_pointers: Vec<usize>,
    row_indices: Vec<usize>,
    symbolic: std::sync::Arc<faer::sparse::linalg::lu::SymbolicLu<usize>>,
}
thread_local! {static STATIONARY_SYMBOLIC:RefCell<VecDeque<StationarySymbolic>>=const{RefCell::new(VecDeque::new())};}
struct StationaryNumeric {
    dimension: usize,
    numeric: faer::sparse::linalg::lu::NumericLu<usize, f64>,
}
thread_local! {static STATIONARY_NUMERIC:RefCell<Option<StationaryNumeric>>=const{RefCell::new(None)};}
struct StationaryOperatorResult {
    matrix: Sparse,
    physical_dimension: usize,
    indices: Vec<usize>,
    frequencies: Vec<f64>,
    harmonics: Vec<(Vec<i32>, bool)>,
    coefficients: Vec<f64>,
    residual: f64,
    boundary_residual: f64,
    trace_error: f64,
    positivity_error: f64,
    condition: f64,
}
thread_local! {static STATIONARY_OPERATORS:RefCell<VecDeque<StationaryOperatorResult>>=const{RefCell::new(VecDeque::new())};}
#[derive(Clone)]
struct Harmonic {
    index: Vec<i32>,
    offset: usize,
    imaginary: bool,
}
struct Extended {
    matrix: Sparse,
    initial: Vec<f64>,
    harmonics: Vec<Harmonic>,
    physical_dimension: usize,
    frequency_basis: FrequencyBasis,
}
fn structured_layout(ext:&Extended,indices:&[usize],weights:&[f64])->crate::structured_floquet::Layout{
    crate::structured_floquet::Layout{
        frequencies:ext.frequency_basis.frequencies.clone(),
        drive_indices:ext.frequency_basis.drive_indices.clone(),
        harmonics:ext.harmonics.iter().map(|h|crate::structured_floquet::HarmonicSpec{index:h.index.clone(),offset:h.offset,imaginary:h.imaginary}).collect(),
        indices:indices.to_vec(),weights:weights.to_vec(),
    }
}
fn refinement_layout(ext:&Extended,indices:&[usize],weights:&[f64])->Result<RefinementLayout,String>{
    let mut positions=vec![usize::MAX;ext.matrix.len()];
    for(local,&global)in indices.iter().enumerate(){if global>=positions.len()||positions[global]!=usize::MAX{return Err("Invalid refinement coordinate indices".into());}positions[global]=local;}
    let mut coordinates:Vec<Option<RefinementCoordinate>>=vec![None;indices.len()];
    for h in &ext.harmonics{for part in 0..if h.imaginary{2}else{1}{for physical in 0..ext.physical_dimension{
        let global=h.offset+part*ext.physical_dimension+physical;let local=positions[global];
        if local!=usize::MAX{coordinates[local]=Some(RefinementCoordinate{harmonic:h.index.clone(),part,physical});}
    }}}
    if weights.len()!=ext.matrix.len(){return Err("Refinement coordinate weights have the wrong dimension".into());}
    Ok(RefinementLayout{coordinates:coordinates.into_iter().collect::<Option<Vec<_>>>().ok_or("Unmapped refinement coordinate")?,
        weights:indices.iter().map(|&i|weights[i]).collect(),frequencies:ext.frequency_basis.frequencies.clone(),
        drive_indices:ext.frequency_basis.drive_indices.clone(),physical_dimension:ext.physical_dimension})
}
fn refinement_embedding(old:&RefinementLayout,new:&RefinementLayout)->Result<Vec<usize>,String>{
    if old.physical_dimension!=new.physical_dimension||old.frequencies!=new.frequencies||old.drive_indices!=new.drive_indices{
        return Err("Refinement preconditioner frequency basis or physical coordinates changed".into());
    }
    if old.coordinates.len()>=new.coordinates.len()||old.weights.len()!=old.coordinates.len()||new.weights.len()!=new.coordinates.len(){return Err("Refinement preconditioner requires a strictly larger coordinate space".into());}
    let positions:HashMap<_,_>=new.coordinates.iter().enumerate().map(|(i,key)|(key,i)).collect();
    if positions.len()!=new.coordinates.len(){return Err("Duplicate refined harmonic coordinate".into());}
    old.coordinates.iter().enumerate().map(|(i,key)|{
        let &j=positions.get(key).ok_or("Refinement omitted a previous harmonic coordinate")?;
        if old.weights[i]!=new.weights[j]{return Err("Refinement changed a retained coordinate weight".into());}Ok(j)
    }).collect()
}
fn structured_stationary_products(model:&Model,ext:&Extended,indices:&[usize],trace:usize)
    ->Result<(crate::block_solve::RawProduct,crate::block_solve::RawProduct),String>{
    let kernel=std::rc::Rc::new(crate::structured_floquet::Prepared::new(model,
        structured_layout(ext,indices,&vec![1.;ext.matrix.len()]))?);
    let population_columns:Vec<_>=indices.iter().enumerate().filter_map(|(j,&global)|(global<model.dimension).then_some(j)).collect();
    let trace_global=indices[trace];
    let row_difference:Vec<_>=indices.iter().map(|&global|{
        let original=ext.matrix[global].iter().filter(|&&(row,_)|row==trace_global).map(|&(_,v)|v).sum::<f64>();
        (if global<model.dimension{1.}else{0.})-original
    }).collect();
    let forward=std::rc::Rc::clone(&kernel);
    Ok((Box::new(move|x:&[f64]|{
        let mut result=forward.product_shifted(x,0.,None);for v in &mut result{*v= -*v;}
        result[trace]=population_columns.iter().map(|&j|x[j]).sum();result
    }),Box::new(move|x:&[f64]|{
        let mut result=kernel.product_shifted_transpose(x,0.,None);
        for(j,value)in result.iter_mut().enumerate(){*value= -*value+row_difference[j]*x[trace];}result
    })))
}

fn canonical(index: &[i32]) -> (Vec<i32>, f64) {
    let sign = index
        .iter()
        .find(|&&x| x != 0)
        .map(|&x| if x > 0 { 1. } else { -1. })
        .unwrap_or(1.);
    (index.iter().map(|&x| x * sign as i32).collect(), sign)
}
fn lattice(basis: &FrequencyBasis, order: i32, max_blocks: usize) -> Result<Vec<Vec<i32>>, String> {
    let zero = vec![0; basis.frequencies.len()];
    let mut values = BTreeSet::from([zero.clone()]);
    let mut frontier = BTreeSet::from([zero]);
    for _ in 0..order {
        let mut next = BTreeSet::new();
        for index in &frontier {
            for drive in &basis.drive_indices {
                for sign in [-1, 1] {
                    let target: Vec<_> = index
                        .iter()
                        .zip(drive)
                        .map(|(&a, &b)| a + sign * b)
                        .collect();
                    if !values.contains(&target) {
                        next.insert(target);
                    }
                }
            }
        }
        values.extend(next.iter().cloned());
        if values.len() > max_blocks {
            return Err("This carrier configuration exceeds the 131072-coordinate harmonic budget. No laser or state was removed.".into());
        }
        frontier = next;
    }
    Ok(values.into_iter().collect())
}
fn compress(matrix: &mut Sparse) {
    for col in matrix {
        col.sort_by_key(|x| x.0);
        let mut out: Vec<(usize, f64)> = Vec::with_capacity(col.len());
        for &(r, v) in col.iter() {
            if let Some(last) = out.last_mut() {
                if last.0 == r {
                    last.1 += v;
                    continue;
                }
            }
            out.push((r, v));
        }
        out.retain(|&(_, v)| v != 0.);
        *col = out;
    }
}
fn hamiltonian_generator(model: &Model, h: &Operator) -> Result<Sparse, String> {
    let mut part = model.clone();
    part.h_real = h.real.clone();
    part.h_imag = h.imag.clone();
    part.collapse.clear();
    part.drives.clear();
    part.time_dependent = false;
    Ok(build_generator(&part)?.matrix)
}

fn extend(model: &Model, base: &Generator, order: i32) -> Result<Extended, String> {
    let d = base.n * base.n;
    let frequencies: Vec<_> = model.drives.iter().map(|d| d.omega).collect();
    let mut frequency_basis = harmonics::model_frequency_basis(model, 16)?;
    if !frequency_basis.exact || frequency_basis.rounded_relations != 0 {
        // Finite-time propagation must retain the actual input frequencies.
        // A low-order relation that holds only after rounding may not replace
        // them; use independent axes rather than changing the Hamiltonian.
        frequency_basis = FrequencyBasis {
            frequencies: frequencies.clone(),
            drive_indices: (0..frequencies.len())
                .map(|i| {
                    (0..frequencies.len())
                        .map(|j| if i == j { 1 } else { 0 })
                        .collect()
                })
                .collect(),
            reconstructed_frequencies: frequencies,
            reconstruction_errors: vec![0.; model.drives.len()],
            exact_relations: 0,
            rounded_relations: 0,
            exact: true,
            search_exhausted: false,
            source_to_basis:None,
        };
    }
    let indices = lattice(&frequency_basis, order, 131072 / d)?;
    let mut harmonics = vec![Harmonic {
        index: vec![0; frequency_basis.frequencies.len()],
        offset: 0,
        imaginary: false,
    }];
    let mut offsets = HashMap::new();
    offsets.insert(vec![0; frequency_basis.frequencies.len()], 0usize);
    let mut size = d;
    for index in &indices {
        if index.iter().all(|&x| x == 0) || canonical(index).1 < 0. {
            continue;
        }
        offsets.insert(index.clone(), size);
        harmonics.push(Harmonic {
            index: index.clone(),
            offset: size,
            imaginary: true,
        });
        size += 2 * d;
    }
    let mut matrix = vec![Vec::new(); size];
    let coefficients: Vec<_> = model
        .drives
        .iter()
        .map(|drive| {
            Ok((
                hamiltonian_generator(model, &drive.cos)?,
                hamiltonian_generator(model, &drive.sin)?,
            ))
        })
        .collect::<Result<_, String>>()?;
    for h in &harmonics {
        let repetitions = if h.imaginary { 2 } else { 1 };
        for part in 0..repetitions {
            for (col, entries) in base.matrix.iter().enumerate() {
                for &(row, v) in entries {
                    add_entry(
                        &mut matrix,
                        h.offset + part * d + row,
                        h.offset + part * d + col,
                        v,
                    );
                }
            }
        }
    }
    // z_n' = (L0 - i n.omega) z_n + (Ac-i As)/2 z_(n-ej)
    //                                    + (Ac+i As)/2 z_(n+ej).
    // z_-n = conj(z_n); the n=0 vector is real in Hermitian coordinates.
    for h in &harmonics {
        if h.imaginary {
            let omega: f64 = h
                .index
                .iter()
                .zip(&frequency_basis.frequencies)
                .map(|(&n, &frequency)| n as f64 * frequency)
                .sum();
            for i in 0..d {
                add_entry(&mut matrix, h.offset + i, h.offset + d + i, omega);
                add_entry(&mut matrix, h.offset + d + i, h.offset + i, -omega);
            }
        }
        for (j, drive) in model.drives.iter().enumerate() {
            if !drive.omega.is_finite() || drive.omega <= 0. {
                return Err("Harmonic angular frequencies must be positive and finite".into());
            }
            let (cosine, sine) = &coefficients[j];
            for direction in [-1, 1] {
                let source: Vec<_> = h
                    .index
                    .iter()
                    .zip(&frequency_basis.drive_indices[j])
                    .map(|(&n, &step)| n + direction * step)
                    .collect();
                let (key, conjugate) = canonical(&source);
                let Some(&input) = offsets.get(&key) else {
                    continue;
                };
                let input_imaginary = source.iter().any(|&n| n != 0);
                // M.re=Ac/2; M.im=direction*As/2.
                for (col, entries) in cosine.iter().enumerate() {
                    for &(row, v) in entries {
                        add_entry(&mut matrix, h.offset + row, input + col, 0.5 * v);
                        if h.imaginary && input_imaginary {
                            add_entry(
                                &mut matrix,
                                h.offset + d + row,
                                input + d + col,
                                0.5 * v * conjugate,
                            );
                        }
                    }
                }
                for (col, entries) in sine.iter().enumerate() {
                    for &(row, v) in entries {
                        let v = 0.5 * direction as f64 * v;
                        if input_imaginary {
                            add_entry(&mut matrix, h.offset + row, input + d + col, -v * conjugate);
                        }
                        if h.imaginary {
                            add_entry(&mut matrix, h.offset + d + row, input + col, v);
                        }
                    }
                }
            }
        }
    }
    compress(&mut matrix);
    let mut initial = vec![0.; size];
    initial[..d].copy_from_slice(&base.initial);
    Ok(Extended {
        matrix,
        initial,
        harmonics,
        physical_dimension: d,
        frequency_basis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rotating_two_level(omega: f64) -> Model {
        Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: vec![0., 0., 0., omega + 1.3],
            h_imag: vec![0.; 4],
            collapse: vec![Operator {
                real: vec![0., 1., 0., 0.],
                imag: vec![0.; 4],
            }],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![Drive {
                omega,
                cos: Operator {
                    real: vec![0., 0.7, 0.7, 0.],
                    imag: vec![0.; 4],
                },
                sin: Operator {
                    real: vec![0.; 4],
                    imag: vec![0., 0.7, -0.7, 0.],
                },
            }],
        }
    }
    #[test]
    fn structured_floquet_product_matches_actual_extended_operator_and_complex_poles(){
        for multiple in [false,true]{
            let mut model=rotating_two_level(7.);
            model.h_real[1]=0.13;model.h_real[2]=0.13;model.h_imag[1]=0.21;model.h_imag[2]=-0.21;
            if multiple{let mut drive=model.drives[0].clone();drive.omega=2_f64.sqrt();
                for x in &mut drive.cos.real{*x*=0.37;}for x in &mut drive.sin.imag{*x*= -0.23;}model.drives.push(drive);}
            let g=build_generator(&model).unwrap();let ext=extend(&model,&g,2).unwrap();
            let mut weights=vec![1.;ext.matrix.len()];
            for h in &ext.harmonics{for part in 0..if h.imaginary{2}else{1}{for i in 0..ext.physical_dimension{
                weights[h.offset+part*ext.physical_dimension+i]=((if i<model.dimension{1_f64}else{2.})*(if h.imaginary{2.}else{1.})).sqrt();
            }}}
            for subset in [false,true]{
                let indices:Vec<_>=(0..ext.matrix.len()).rev().filter(|i|!subset||i%3!=1).collect();let r=indices.len();
                let mut positions=vec![usize::MAX;ext.matrix.len()];for(i,&global)in indices.iter().enumerate(){positions[global]=i;}
                let mut exact=vec![Vec::new();r];
                for(j,&global)in indices.iter().enumerate(){for &(row,value)in &ext.matrix[global]{if positions[row]!=usize::MAX{
                    exact[j].push((positions[row],value*weights[row]/weights[global]));
                }}}
                let prepared=crate::structured_floquet::Prepared::new(&model,structured_layout(&ext,&indices,&weights)).unwrap();
                for frequency in [None,Some(0.47),Some(-1.25)]{for phase in [0.2,0.9]{
                    let x:Vec<_>=(0..r*if frequency.is_some(){2}else{1}).map(|i|(i as f64*0.731+phase).sin()).collect();
                    let mut expected:Vec<_>=x.iter().map(|v|0.31*v).collect();
                    for part in 0..if frequency.is_some(){2}else{1}{for(j,column)in exact.iter().enumerate(){for &(i,value)in column{
                        expected[part*r+i]-=value*x[part*r+j];
                    }}}
                    if let Some(frequency)=frequency{for i in 0..r{expected[i]-=frequency*x[r+i];expected[r+i]+=frequency*x[i];}}
                    let actual=prepared.try_product_shifted(&x,0.31,frequency).unwrap();
                    let scale=expected.iter().map(|v|v.abs()).fold(1.,f64::max);
                    let error=actual.iter().zip(expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
                    assert!(error<2e-12*scale,"multiple={multiple} subset={subset} frequency={frequency:?}: {error:e}");
                    let mut transpose:Vec<_>=x.iter().map(|v|0.31*v).collect();
                    for part in 0..if frequency.is_some(){2}else{1}{for(j,column)in exact.iter().enumerate(){for &(i,value)in column{
                        transpose[part*r+j]-=value*x[part*r+i];
                    }}}
                    if let Some(frequency)=frequency{for i in 0..r{transpose[i]+=frequency*x[r+i];transpose[r+i]-=frequency*x[i];}}
                    let actual=prepared.try_product_shifted_transpose(&x,0.31,frequency).unwrap();
                    let scale=transpose.iter().map(|v|v.abs()).fold(1.,f64::max);
                    let error=actual.iter().zip(transpose).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
                    assert!(error<2e-12*scale,"transpose multiple={multiple} subset={subset} frequency={frequency:?}: {error:e}");
                }}
            }
        }
    }
    #[test]
    fn structured_stationary_trace_row_and_its_transpose_match_explicit_system(){
        let model=rotating_two_level(3.7);let g=build_generator(&model).unwrap();let ext=extend(&model,&g,2).unwrap();
        let indices:Vec<_>=(0..ext.matrix.len()).rev().collect();let r=indices.len();
        let mut positions=vec![0;r];for(i,&j)in indices.iter().enumerate(){positions[j]=i;}
        let trace=indices.iter().position(|&i|i<model.dimension).unwrap();
        let (forward,transpose)=structured_stationary_products(&model,&ext,&indices,trace).unwrap();
        for phase in [0.3,0.8]{
            let x:Vec<_>=(0..r).map(|i|(i as f64*0.713+phase).cos()).collect();
            let mut expected=vec![0.;r];let mut expected_t=vec![0.;r];
            for(j,&global)in indices.iter().enumerate(){
                for &(row,value)in &ext.matrix[global]{let i=positions[row];if i!=trace{expected[i]+=value*x[j];expected_t[j]+=value*x[i];}}
                if global<model.dimension{expected[trace]+=x[j];expected_t[j]+=x[trace];}
            }
            for(actual,expected)in[(forward(&x),expected),(transpose(&x),expected_t)]{
                let scale=expected.iter().map(|v|v.abs()).fold(1.,f64::max);
                assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-12*scale));
            }
        }
    }
    #[test]
    fn deferred_stationary_condition_keeps_the_same_final_gate_and_coefficients() {
        let lap = Lapack::load().unwrap();
        let model = rotating_two_level(7.);
        let g = build_generator(&model).unwrap();
        clear_request_results();
        let result = (|| -> Result<(), String> {
            let mut candidate = stationary_candidate(&lap, &model, &g, 2, false)?;
            assert!(candidate.pending_condition.is_some());
            assert!(!candidate.condition.is_finite());
            assert!(STATIONARY_OPERATORS.with(|cache| cache.borrow().is_empty()),
                "An uncertified candidate must never enter the proof cache");
            let coefficients = candidate.coefficients.clone();
            certify_stationary_candidate(&lap, &mut candidate, 2)?;
            assert!(candidate.pending_condition.is_none());
            assert!(candidate.condition.is_finite() && candidate.condition < 1e12);
            assert_eq!(candidate.coefficients, coefficients);
            let reference = stationary(&lap, &model, &g, 2)?;
            assert_eq!(candidate.coefficients, reference.coefficients);
            assert_eq!(candidate.condition, reference.condition);
            Ok(())
        })();
        clear_request_results();
        result.unwrap();
    }
    #[test]
    fn deferred_condition_still_rejects_an_unresolved_global_system() {
        let lap = Lapack::load().unwrap();
        let matrix = vec![vec![(0, 1.), (1, 1.)], vec![(0, 1.), (1, 1. + 1e-13)]];
        // Well-conditioned individual diagonal blocks do not establish a
        // well-conditioned original system. Its unchanged global gate fails.
        let prepared = crate::block_solve::Prepared::new(&lap, &matrix, &[], 10.).unwrap();
        assert!(block_condition_estimate(&lap, &prepared).is_err());
    }
    #[test]
    fn periodic_rotation_matches_exact_optical_bloch_population() {
        let lap = Lapack::load().unwrap();
        for omega in [0.2, 7., 1e5] {
            let m = rotating_two_level(omega);
            let out = solve(
                &lap,
                &m,
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            )
            .unwrap();
            let expected = 0.7_f64.powi(2) / (1.3_f64.powi(2) + 0.25 + 2. * 0.7_f64.powi(2));
            assert!(
                (out["asymptoticPopulations"][1].as_f64().unwrap() - expected).abs() < 2e-11,
                "omega={omega}, output={out}"
            );
            assert_eq!(out["asymptoticStatus"], "initial-dependent-mean");
        }
    }
    #[test]
    fn extended_generator_reconstructs_exact_initial_derivative() {
        let m = rotating_two_level(9.);
        let g = build_generator(&m).unwrap();
        let ext = extend(&m, &g, 1).unwrap();
        let derivative = sparse_product(&ext.matrix, &ext.initial);
        let actual = reconstruct(&ext, &derivative, &[0.]);
        let mut at_zero = m.clone();
        for i in 0..4 {
            at_zero.h_real[i] += m.drives[0].cos.real[i];
            at_zero.h_imag[i] += m.drives[0].cos.imag[i];
        }
        let direct = build_generator(&at_zero).unwrap();
        let expected = sparse_product(&direct.matrix, &g.initial);
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-14);
        }
    }
    #[test]
    fn nonunique_periodic_mean_retains_initial_and_decay_branching() {
        let fixtures: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/floquet-independent-cases.json"
        ))
        .unwrap();
        let lap = Lapack::load().unwrap();
        for case in fixtures["cases"].as_array().unwrap().iter().filter(|case| {
            matches!(
                case["name"].as_str(),
                Some("nonunique-constant-populations" | "decay-to-two-dark-states")
            )
        }) {
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let out = solve(
                &lap,
                &model,
                &Options {
                    steady_only: true,
                    ..Options::default()
                },
            )
            .unwrap();
            assert_eq!(out["asymptoticStatus"], "initial-dependent-mean");
            assert!((out["asymptoticPopulations"][0].as_f64().unwrap() - 0.3).abs() < 1e-10);
            assert!((out["asymptoticPopulations"][1].as_f64().unwrap() - 0.7).abs() < 1e-10);
        }
    }
    #[test]
    fn rounded_frequency_relation_never_becomes_an_infinite_time_claim() {
        let op = |v: f64| Operator {
            real: vec![v, 0., 0., -v],
            imag: vec![0.; 4],
        };
        let model = Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: vec![0., 0.2, 0.2, 0.],
            h_imag: vec![0.; 4],
            collapse: vec![Operator {
                real: vec![0., 0.8_f64.sqrt(), 0., 0.],
                imag: vec![0.; 4],
            }],
            initial_index: 1,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![
                Drive {
                    omega: 0.1,
                    cos: op(0.001),
                    sin: op(0.),
                },
                Drive {
                    omega: 3. * 0.1,
                    cos: op(0.002),
                    sin: op(0.),
                },
            ],
        };
        let lap = Lapack::load().unwrap();
        let g = build_generator(&model).unwrap();
        let ext = extend(&model, &g, 1).unwrap();
        assert_eq!(ext.frequency_basis.frequencies, vec![0.1, 3. * 0.1]);
        assert_eq!(
            ext.frequency_basis.drive_indices,
            vec![vec![1, 0], vec![0, 1]]
        );
        assert!(solve(
            &lap,
            &model,
            &Options {
                steady_only: true,
                ..Options::default()
            }
        )
        .unwrap_err()
        .contains("rounding"));
    }
    #[test]
    fn failed_stationary_factor_drops_workspace_before_next_valid_solve() {
        initialize_numeric_parallelism();
        STATIONARY_NUMERIC.with(|workspace| *workspace.borrow_mut() = None);
        let solve = |entries: Vec<Triplet<usize, usize, f64>>| {
            let a = SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &entries).unwrap();
            let symbolic = faer::sparse::linalg::lu::factorize_symbolic_lu(
                a.as_ref().symbolic(),
                Default::default(),
            )
            .unwrap();
            stationary_linear_solve(&symbolic, a.as_ref(), 0, 2.)
        };
        let first = solve(vec![Triplet::new(0, 0, 1.), Triplet::new(1, 1, 1.)]).unwrap();
        assert_eq!(first.values, vec![1., 0.]);
        assert!(STATIONARY_NUMERIC.with(|workspace| workspace.borrow().is_some()));
        let failed = solve(vec![
            Triplet::new(0, 0, 1.),
            Triplet::new(1, 0, 1.),
            Triplet::new(0, 1, 1.),
            Triplet::new(1, 1, 1.),
        ]);
        assert!(failed.is_err());
        assert!(STATIONARY_NUMERIC.with(|workspace| workspace.borrow().is_none()));
        let last = solve(vec![
            Triplet::new(0, 0, 1.),
            Triplet::new(1, 0, -0.5),
            Triplet::new(1, 1, 1.),
        ])
        .unwrap();
        assert_eq!(last.values, vec![1., 0.5]);
        assert!(!last.workspace_reused);
    }
    #[test]
    fn stationary_workspace_never_reuses_a_different_matrix_dimension() {
        initialize_numeric_parallelism();
        STATIONARY_NUMERIC.with(|workspace| *workspace.borrow_mut() = None);
        for (n, reused) in [(3, false), (2, false), (3, true)] {
            let entries: Vec<_> = (0..n).map(|i| Triplet::new(i, i, 1.)).collect();
            let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &entries).unwrap();
            let symbolic = faer::sparse::linalg::lu::factorize_symbolic_lu(
                a.as_ref().symbolic(),
                Default::default(),
            )
            .unwrap();
            let result = stationary_linear_solve(&symbolic, a.as_ref(), 0, 1.).unwrap();
            assert_eq!(result.workspace_reused, reused);
            assert_eq!(result.values[0], 1.);
            assert!(result.values[1..].iter().all(|&x| x == 0.));
        }
    }
}

fn sparse_product(a: &Sparse, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.; a.len()];
    for (col, entries) in a.iter().enumerate() {
        for &(row, v) in entries {
            out[row] += v * x[col];
        }
    }
    out
}
fn reconstruct(extended: &Extended, x: &[f64], phases: &[f64]) -> Vec<f64> {
    let d = extended.physical_dimension;
    let mut out = x[..d].to_vec();
    for h in extended.harmonics.iter().skip(1) {
        let phase: f64 = h
            .index
            .iter()
            .zip(phases)
            .map(|(&n, &p)| n as f64 * p)
            .sum();
        let (s, c) = phase.sin_cos();
        for i in 0..d {
            out[i] += 2. * (c * x[h.offset + i] - s * x[h.offset + d + i]);
        }
    }
    out
}
struct Stationary {
    extended: Extended,
    coefficients: Vec<f64>,
    residual: f64,
    boundary_residual: f64,
    trace_error: f64,
    positivity_error: f64,
    dimension: usize,
    condition: f64,
    valid: bool,
    proof_cached: bool,
    symbolic_cached: bool,
    // Intermediate harmonic candidates are never published or proof-cached.
    // Keep their factor only until the cheap convergence gates decide whether
    // this order needs the unchanged full condition estimate.
    pending_condition: Option<std::rc::Rc<crate::block_solve::Prepared>>,
    prepared:Option<std::rc::Rc<crate::block_solve::Prepared>>,
}
fn fourier_coefficients(ext: &Extended, x: &[f64]) -> Vec<FourierCoefficient> {
    let d = ext.physical_dimension;
    ext.harmonics
        .iter()
        .map(|h| FourierCoefficient {
            index: h.index.clone(),
            real: x[h.offset..h.offset + d].to_vec(),
            imag: if h.imaginary {
                x[h.offset + d..h.offset + 2 * d].to_vec()
            } else {
                vec![0.; d]
            },
        })
        .collect()
}
fn fourier_density(g: &Generator, ext: &Extended, coefficients: &[f64]) -> Value {
    let harmonics:Vec<_>=fourier_coefficients(ext,coefficients).into_iter().map(|h|{let mut rho=vec![C::default();g.n*g.n];for(k,coord)in g.coords.iter().enumerate(){let z=C::new(h.real[k],h.imag[k]);match *coord{Coordinate::Population(i)=>rho[i*g.n+i]+=z,Coordinate::Real(i,j)=>{rho[i*g.n+j]+=z;rho[j*g.n+i]+=z;},Coordinate::Imag(i,j)=>{let iz=C::new(-z.im,z.re);rho[i*g.n+j]+=iz;rho[j*g.n+i]+=iz*-1.;}}}json!({"index":h.index,"real":rho.iter().map(|z|z.re).collect::<Vec<_>>(),"imag":rho.iter().map(|z|z.im).collect::<Vec<_>>()})}).collect();
    let mut result=json!({"frequenciesRadPerUs":ext.frequency_basis.frequencies,"harmonics":harmonics,
        "frequencyCoordinateSource":ext.frequency_basis.source()});
    if let Some(mapping)=&ext.frequency_basis.source_to_basis{result["sourceToBasis"]=json!(mapping);}result
}
fn initial_mean(
    lap: &Lapack,
    model: &Model,
    g: &Generator,
    ext: &Extended,
) -> Result<Option<Vec<f64>>, String> {
    let inds = reachable(&ext.matrix, &ext.initial);
    let r = inds.len();
    if r > 512 {
        return Ok(None);
    }
    let mut pos = vec![usize::MAX; ext.matrix.len()];
    for (i, &global) in inds.iter().enumerate() {
        pos[global] = i;
    }
    let mut a = vec![0.; r * r];
    for (j, &global) in inds.iter().enumerate() {
        for &(row, v) in &ext.matrix[global] {
            if pos[row] != usize::MAX {
                a[pos[row] + j * r] = v;
            }
        }
    }
    let spec = match spectral(lap, &a, r) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let initial: Vec<_> = inds.iter().map(|&i| ext.initial[i]).collect();
    let c = lap.solve_factored(&spec.lu, &spec.pivots, &initial, r)?;
    let tolerance = (16. * spec.residual.max(f64::EPSILON * spec.norm)).max(2e-11);
    if tolerance > 1e-7 || spec.wr.iter().any(|&value| value > tolerance) {
        return Ok(None);
    }
    let mut mean = vec![0.; ext.physical_dimension];
    let mut j = 0;
    while j < r {
        let pair = spec.wi[j] > 0.;
        let advance = if pair { 2 } else { 1 };
        if spec.wr[j].abs() <= tolerance {
            let weight = |global: usize| -> C {
                let i = pos[global];
                if i == usize::MAX {
                    return C::default();
                }
                if pair {
                    C::new(
                        spec.vectors[i + j * r] * c[j] + spec.vectors[i + (j + 1) * r] * c[j + 1],
                        spec.vectors[i + (j + 1) * r] * c[j] - spec.vectors[i + j * r] * c[j + 1],
                    )
                } else {
                    C::new(spec.vectors[i + j * r] * c[j], 0.)
                }
            };
            for h in &ext.harmonics {
                let omega: f64 = h
                    .index
                    .iter()
                    .zip(&ext.frequency_basis.frequencies)
                    .map(|(&n, &f)| n as f64 * f)
                    .sum();
                if !h.imaginary {
                    if spec.wi[j].abs() <= tolerance {
                        for (i, value) in mean.iter_mut().enumerate() {
                            *value += weight(i).re;
                        }
                    }
                } else if pair {
                    for (i, value) in mean.iter_mut().enumerate() {
                        let u = weight(h.offset + i);
                        let v = weight(h.offset + ext.physical_dimension + i);
                        if (spec.wi[j] + omega).abs() <= tolerance {
                            *value += u.re - v.im;
                        }
                        if (spec.wi[j] - omega).abs() <= tolerance {
                            *value += u.re + v.im;
                        }
                    }
                } else if omega == 0. {
                    for (i, value) in mean.iter_mut().enumerate() {
                        *value += 2. * weight(h.offset + i).re;
                    }
                }
            }
        }
        j += advance;
    }
    // This is the temporal mean of the physical reconstructed density. In a
    // nonunique Floquet space, keeping only K's zero modes would average away
    // initial optical phase: lambda = -i n.omega modes also contribute here.
    if mean.iter().any(|x| !x.is_finite())
        || (mean[..model.dimension].iter().sum::<f64>() - 1.).abs() > 1e-8
        || positivity(lap, &expand_density(&g.coords, &mean, g.n), g.n)? > 1e-8
    {
        return Ok(None);
    }
    Ok(Some(mean))
}
fn phase_sample(dimensions: usize, sample: usize) -> Vec<f64> {
    const PRIMES: [usize; 8] = [2, 3, 5, 7, 11, 13, 17, 19];
    (0..dimensions)
        .map(|j| {
            let radix = PRIMES[j % PRIMES.len()];
            let mut value = sample + 1;
            let mut fraction = 1.;
            let mut phase = 0.;
            while value > 0 {
                fraction /= radix as f64;
                phase += (value % radix) as f64 * fraction;
                value /= radix;
            }
            phase * 2. * std::f64::consts::PI
        })
        .collect()
}
fn inverse_one_norm_with(n: usize, mut solve: impl FnMut(&Mat<f64>, bool) -> Mat<f64>) -> f64 {
    // Hager/Higham inverse one-norm estimate using solves and transpose solves.
    // It detects near-singular stationary systems without forming an inverse.
    let mut x = Mat::<f64>::from_fn(n, 1, |_, _| 1. / n as f64);
    let mut estimate = 0_f64;
    let mut previous = usize::MAX;
    for _ in 0..8 {
        let y = solve(&x, false);
        if (0..n).any(|i| !y[(i, 0)].is_finite()) {
            return f64::INFINITY;
        }
        estimate = estimate.max((0..n).map(|i| y[(i, 0)].abs()).sum());
        let signs = Mat::<f64>::from_fn(n, 1, |i, _| if y[(i, 0)] < 0. { -1. } else { 1. });
        let z = solve(&signs, true);
        if (0..n).any(|i| !z[(i, 0)].is_finite()) {
            return f64::INFINITY;
        }
        let j = (0..n)
            .max_by(|&i, &j| z[(i, 0)].abs().total_cmp(&z[(j, 0)].abs()))
            .unwrap_or(0);
        let projection: f64 = (0..n).map(|i| z[(i, 0)] * x[(i, 0)]).sum();
        if j == previous || z[(j, 0)].abs() <= projection {
            break;
        }
        previous = j;
        x = Mat::from_fn(n, 1, |i, _| if i == j { 1. } else { 0. });
    }
    let alternating = Mat::<f64>::from_fn(n, 1, |i, _| {
        (if i % 2 == 0 { 1. } else { -1. }) * (1. + i as f64 / (n - 1).max(1) as f64)
    });
    let y = solve(&alternating, false);
    if (0..n).any(|i| !y[(i, 0)].is_finite()) {
        return f64::INFINITY;
    }
    estimate.max(2. * (0..n).map(|i| y[(i, 0)].abs()).sum::<f64>() / (3. * n as f64))
}
pub(super) struct StationaryLinearSolution {
    pub values: Vec<f64>,
    pub condition: f64,
    pub numeric_ms: f64,
    pub condition_ms: f64,
    pub solve_ms: f64,
    pub workspace_reused: bool,
    pending_condition: Option<std::rc::Rc<crate::block_solve::Prepared>>,
    prepared:Option<std::rc::Rc<crate::block_solve::Prepared>>,
}
fn block_condition_estimate(lap: &Lapack, prepared: &crate::block_solve::Prepared) -> Result<f64, String> {
    let n = prepared.matrix.len();
    let mut failed = false;
    let mut solve = |rhs: &Mat<f64>, transposed: bool| {
        let b: Vec<_> = (0..n)
            .map(|i| rhs[(i, 0)] * if transposed { 1. } else { prepared.scales[i] })
            .collect();
        let result = if transposed {
            prepared.solve_scaled_transpose(lap, &b)
        } else {
            prepared.solve(lap, &b)
        };
        match result {
            Ok(x) => Mat::from_fn(n, 1, |i, _| x[i]),
            Err(_) => {
                failed = true;
                Mat::from_fn(n, 1, |_, _| f64::NAN)
            }
        }
    };
    let operator_norm=prepared.operator_norm(false);
    let condition = operator_norm * inverse_one_norm_with(n, &mut solve);
    if failed || !condition.is_finite() || condition > 1e12 {
        return Err("Block stationary condition unresolved".into());
    }
    Ok(condition)
}
fn block_stationary_solve(
    lap: &Lapack,
    entries: &[(usize, usize, f64)],
    n: usize,
    trace: usize,
    central: &[usize],
    omega: f64,
    require_condition: bool,
    products:Option<(crate::block_solve::RawProduct,crate::block_solve::RawProduct)>,
    layout:RefinementLayout,
) -> Result<StationaryLinearSolution, String> {
    let started = Instant::now();
    let mut a = vec![Vec::new(); n];
    for &(i, j, v) in entries {
        a[j].push((i, v));
    }
    compress(&mut a);
    let extended=MEAN_REFINEMENT_PRECONDITIONER.with(|cache|{
        let entry=cache.borrow();let previous=entry.as_ref()?;
        let attempt=(||{
            if layout.coordinates.get(trace)!=Some(&previous.trace){return Err("Stationary normalization row changed".into());}
            let embedding=refinement_embedding(&previous.layout,&layout)?;
            let factor=previous.prepared.raw_preconditioner().ok_or("Previous stationary factor is unavailable or retired")?;
            crate::block_solve::Prepared::extend_raw(lap,&a,&previous.matrix,factor,&embedding,0.02*omega)
        })();
        match attempt{
            Ok(prepared)=>Some(prepared),
            Err(_)=>None,
        }
    });
    let prepared=if let Some(prepared)=extended{prepared}else{crate::block_solve::Prepared::new(lap, &a, central, 0.02 * omega)?};
    let prepared=std::rc::Rc::new(if let Some((forward,transpose))=products{
        prepared.with_raw_product(forward).with_raw_transpose(transpose)
    }else{prepared});
    let numeric_ms = started.elapsed().as_secs_f64() * 1000.;
    let before = Instant::now();
    let condition = if require_condition { block_condition_estimate(lap, &prepared)? } else { f64::NAN };
    let condition_ms = before.elapsed().as_secs_f64() * 1000.;
    let before = Instant::now();
    let mut rhs = vec![0.; n];
    rhs[trace] = 1.;
    let values = prepared.solve(lap, &rhs)?;
    if prepared.raw_preconditioner().is_some(){
        let trace=layout.coordinates.get(trace).ok_or("Stationary normalization coordinate is missing")?.clone();
        MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=Some(ReusableStationary{
            layout,trace,matrix:a,prepared:std::rc::Rc::clone(&prepared),
        }));
    }
    Ok(StationaryLinearSolution {
        values,
        condition,
        numeric_ms,
        condition_ms,
        solve_ms: before.elapsed().as_secs_f64() * 1000.,
        workspace_reused: false,
        pending_condition: (!require_condition).then(||std::rc::Rc::clone(&prepared)),
        prepared:Some(prepared),
    })
}
pub(super) fn stationary_linear_solve(
    symbolic: &faer::sparse::linalg::lu::SymbolicLu<usize>,
    a: faer::sparse::SparseColMatRef<'_, usize, f64>,
    trace: usize,
    operator_norm: f64,
) -> Result<StationaryLinearSolution, String> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    let n = a.nrows();
    let keep_workspace = n <= 8192;
    // Keep at most one allocation, for the largest ordinary dimension seen.
    // faer's simplicial NumericLu cannot safely shrink its CSC column arrays,
    // so only SAME-dimension factors reuse storage. Smaller/larger exceptional
    // systems use temporary objects, even when the physical support is related.
    let saved = if keep_workspace {
        STATIONARY_NUMERIC.with(|value| {
            let mut value = value.borrow_mut();
            if value.as_ref().is_some_and(|entry| entry.dimension == n) {
                value.take()
            } else {
                None
            }
        })
    } else {
        None
    };
    let workspace_reused = saved.is_some();
    let mut numeric = saved
        .map(|entry| entry.numeric)
        .unwrap_or_else(faer::sparse::linalg::lu::NumericLu::new);
    let result: Result<StationaryLinearSolution, String> = (|| {
        let par = faer::get_global_parallelism();
        let before = Instant::now();
        let mut factor_memory = MemBuffer::try_new(
            symbolic.factorize_numeric_lu_scratch::<f64>(par, Default::default()),
        )
        .map_err(|e| format!("Periodic LU workspace allocation: {e:?}"))?;
        let lu = symbolic
            .factorize_numeric_lu(
                &mut numeric,
                a,
                par,
                MemStack::new(&mut factor_memory),
                Default::default(),
            )
            .map_err(|e| format!("Periodic stationary LU failed (possibly nonunique): {e:?}"))?;
        let numeric_ms = before.elapsed().as_secs_f64() * 1000.;
        let before = Instant::now();
        let mut solve_memory =
            MemBuffer::try_new(symbolic.solve_in_place_scratch::<f64>(1, par))
                .map_err(|e| format!("Periodic solve workspace allocation: {e:?}"))?;
        let mut solve = |rhs: &Mat<f64>, transpose: bool| {
            let mut x = rhs.clone();
            if transpose {
                lu.solve_transpose_in_place_with_conj(
                    faer::Conj::No,
                    x.as_mut(),
                    par,
                    MemStack::new(&mut solve_memory),
                );
            } else {
                lu.solve_in_place_with_conj(
                    faer::Conj::No,
                    x.as_mut(),
                    par,
                    MemStack::new(&mut solve_memory),
                );
            }
            x
        };
        let condition = operator_norm * inverse_one_norm_with(n, &mut solve);
        let condition_ms = before.elapsed().as_secs_f64() * 1000.;
        if !condition.is_finite() || condition > 1e12 {
            return Err(format!("Periodic stationary system is numerically singular or unresolved (estimated condition {condition:e}); initial dependence cannot be discarded"));
        }
        let before = Instant::now();
        let rhs = Mat::<f64>::from_fn(n, 1, |i, _| if i == trace { 1. } else { 0. });
        let x = solve(&rhs, false);
        let values: Vec<_> = (0..n).map(|i| x[(i, 0)]).collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite periodic stationary solution".into());
        }
        Ok(StationaryLinearSolution {
            values,
            condition,
            numeric_ms,
            condition_ms,
            solve_ms: before.elapsed().as_secs_f64() * 1000.,
            workspace_reused,
            pending_condition: None,
            prepared:None,
        })
    })();
    // Never retain a partially written failed/singular factor. A later valid
    // request starts with clean storage; the symbolic cache remains independent.
    if keep_workspace && result.is_ok() {
        STATIONARY_NUMERIC.with(|value| {
            let mut value = value.borrow_mut();
            if value.as_ref().is_none_or(|entry| entry.dimension <= n) {
                *value = Some(StationaryNumeric {
                    dimension: n,
                    numeric,
                });
            }
        });
    }
    result
}

#[cfg(test)]
mod stationary_refinement_tests{
    use super::*;
    fn driven_two_level()->Model{
        Model{ frequency_coordinates:None,dimension:2,states:vec![],h_real:vec![0.,0.3,0.3,1.2],h_imag:vec![0.,0.1,-0.1,0.],
            collapse:vec![Operator{real:vec![0.,0.7,0.,0.],imag:vec![0.;4]}],
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:true,
            drives:vec![Drive{omega:2.3,cos:Operator{real:vec![0.,0.2,0.2,0.],imag:vec![0.;4]},
                sin:Operator{real:vec![0.;4],imag:vec![0.,0.15,-0.15,0.]}}]}
    }
    #[test]
    fn stationary_raw_correction_resolves_error_hidden_by_row_equilibration(){
        let lap=Lapack::load().unwrap();let a=vec![vec![(0,1.)],vec![(1,1e12)]];
        let prepared=crate::block_solve::Prepared::new(&lap,&a,&[0,1],0.).unwrap();
        let rhs=[1.,1.];let mut x=[1.,1e-12+1e-15];
        let balanced_rhs:Vec<_>=rhs.iter().zip(&prepared.scales).map(|(b,s)|b/s).collect();
        let balanced=sparse_product(&prepared.matrix,&x);
        let balanced_error=balanced_rhs.iter().zip(balanced).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
        assert!(balanced_error<1e-11); // original balanced acceptance is unchanged
        let(before,after)=stationary_raw_correction(&lap,&prepared,&a,&rhs,&mut x).unwrap();
        assert!(before>1e-5);assert!(after<1e-10);assert_eq!(x[0],1.);
        assert!((x[1]-1e-12).abs()<1e-26);
    }
    #[test]
    fn same_order_stationary_refinement_recovers_density_without_clipping_or_renormalizing(){
        let lap=Lapack::load().unwrap();let mut model=driven_two_level();model.drives.clear();model.time_dependent=false;
        let g=build_generator(&model).unwrap();let mut candidate=stationary(&lap,&model,&g,1).unwrap();
        let expected=candidate.coefficients.clone();let original_harmonics=candidate.extended.harmonics.len();
        candidate.coefficients[0]+=2e-4;candidate.coefficients[2]-=1e-4;
        refresh_stationary_checks(&lap,&model,&g,&mut candidate).unwrap();assert!(!stationary_numerics_pass(&candidate));
        refine_stationary_candidate(&lap,&model,&g,&mut candidate,1).unwrap();
        assert!(stationary_numerics_pass(&candidate));assert_eq!(candidate.extended.harmonics.len(),original_harmonics);
        assert!(candidate.coefficients.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));
        assert!(candidate.pending_condition.is_some());certify_stationary_candidate(&lap,&mut candidate,1).unwrap();
        assert!(candidate.condition.is_finite());
    }
    #[test]
    fn psd_only_stationary_failure_returns_to_harmonic_refinement(){
        let model=driven_two_level();let g=build_generator(&model).unwrap();
        let mut candidate=unresolved_stationary(extend(&model,&g,1).unwrap());candidate.valid=true;
        candidate.residual=5e-6;candidate.trace_error=5e-9;candidate.positivity_error=2e-8;
        assert!(stationary_linear_pass(&candidate));assert!(!stationary_numerics_pass(&candidate));
        assert!(!stationary_requires_correction(&candidate,0.,0.));
        candidate.residual=1.1e-5;assert!(stationary_requires_correction(&candidate,0.,0.));
        candidate.residual=5e-6;candidate.trace_error=1.1e-8;assert!(stationary_requires_correction(&candidate,0.,0.));
        candidate.boundary_residual=2e-7;assert!(!stationary_requires_correction(&candidate,0.,0.));
    }
    #[test]
    fn fourier_output_maps_declared_source_axes_and_omits_absent_legacy_mapping(){
        let mut model=driven_two_level();
        model.frequency_coordinates=Some(FrequencyCoordinates{frequencies:vec![2.3,4.6],frequencies_mhz:None,
            drive_indices:vec![vec![1,0]],frame_indices:vec![vec![0,0],vec![1,-1]],frame_sectors:vec![0,1]});
        let g=build_generator(&model).unwrap();let ext=extend(&model,&g,1).unwrap();
        let value=fourier_density(&g,&ext,&vec![0.;ext.matrix.len()]);
        assert_eq!(value["sourceToBasis"],json!([[1],[2]]));
        assert_eq!(value["frequencyCoordinateSource"],"declared-carrier-coordinates");
        model.frequency_coordinates=None;let ext=extend(&model,&g,1).unwrap();
        let legacy=fourier_density(&g,&ext,&vec![0.;ext.matrix.len()]);assert!(legacy.get("sourceToBasis").is_none());
    }
    #[test]
    fn refinement_embedding_handles_relocation_new_old_harmonic_coordinates_and_rejects_changed_metadata(){
        let mut model=driven_two_level();let mut second=model.drives[0].clone();second.omega=2.3_f64.sqrt();model.drives.push(second);
        let g=build_generator(&model).unwrap();let old=extend(&model,&g,1).unwrap();let new=extend(&model,&g,2).unwrap();
        let old_indices:Vec<_>=(0..old.matrix.len()).filter(|&i|i!=1).collect();let new_indices:Vec<_>=(0..new.matrix.len()).collect();
        let old_layout=refinement_layout(&old,&old_indices,&vec![1.;old.matrix.len()]).unwrap();
        let new_layout=refinement_layout(&new,&new_indices,&vec![1.;new.matrix.len()]).unwrap();
        let map=refinement_embedding(&old_layout,&new_layout).unwrap();
        assert!(map.iter().zip(&old_indices).any(|(new,old)|new!=old));
        assert!(!map.contains(&1)); // new reachable coordinate in the OLD central harmonic
        for(i,&j)in map.iter().enumerate(){assert_eq!(old_layout.coordinates[i],new_layout.coordinates[j]);}
        let mut changed=new_layout.clone();changed.weights[map[0]]=2.;
        assert!(refinement_embedding(&old_layout,&changed).is_err());
        let mut changed=new_layout;changed.frequencies[0]+=1e-12;
        assert!(refinement_embedding(&old_layout,&changed).is_err());
    }
}

fn stationary(
    lap: &Lapack,
    model: &Model,
    g: &Generator,
    order: i32,
) -> Result<Stationary, String> {
    stationary_candidate(lap, model, g, order, true)
}
fn stationary_candidate(
    lap: &Lapack,
    model: &Model,
    g: &Generator,
    order: i32,
    require_condition: bool,
) -> Result<Stationary, String> {
    let started = Instant::now();
    let mut phase = Instant::now();
    let mut timings = serde_json::Map::new();
    let extended = extend(model, g, order)?;
    // This is an exact invariant coordinate space seeded by the specified
    // density. Only algebraic zero paths are absent. We compute its phase
    // average, without asserting uniqueness outside that initial orbit.
    let inds = reachable(&extended.matrix, &extended.initial);
    let r = inds.len();
    timings.insert(
        "extendAndSupportMs".into(),
        json!(phase.elapsed().as_secs_f64() * 1000.),
    );
    phase = Instant::now();
    let harmonic_key: Vec<_> = extended
        .harmonics
        .iter()
        .map(|h| (h.index.clone(), h.imaginary))
        .collect();
    let cached = STATIONARY_OPERATORS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let i = cache.iter().position(|entry| {
            entry.physical_dimension == g.n
                && entry.indices == inds
                && entry.frequencies == extended.frequency_basis.frequencies
                && entry.harmonics == harmonic_key
                && entry.matrix == extended.matrix
        });
        i.map(|i| cache.remove(i).unwrap())
    });
    if let Some(entry) = cached {
        // The proof covers the complete operator algebra on this exact
        // physical support, so it is independent of the initial density.
        // Keep the NEW Extended.initial; all initial/scattering readouts are
        // reconstructed by the caller from the current Model and Generator.
        let result = Stationary {
            coefficients: entry.coefficients.clone(),
            residual: entry.residual,
            boundary_residual: entry.boundary_residual,
            trace_error: entry.trace_error,
            positivity_error: entry.positivity_error,
            dimension: r,
            condition: entry.condition,
            valid: true,
            proof_cached: true,
            symbolic_cached: false,
            pending_condition: None,
            prepared:None,
            extended,
        };
        STATIONARY_PROFILE.with(|profile|{if let Some(records)=profile.borrow_mut().as_mut(){records.push(json!({"order":order,"dimension":r,"proofCached":true,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"timings":timings,"conditionEstimate":entry.condition,"residual":entry.residual}));}});
        STATIONARY_OPERATORS.with(|cache| cache.borrow_mut().push_back(entry));
        return Ok(result);
    }
    let mut pos = vec![usize::MAX; extended.matrix.len()];
    for (i, &global) in inds.iter().enumerate() {
        pos[global] = i;
    }
    let trace = inds
        .iter()
        .position(|&i| i < g.n)
        .ok_or("No central population coordinate")?;
    let mut entries = vec![];
    let mut scale = vec![0_f64; r];
    for (j, &global) in inds.iter().enumerate() {
        for &(row, v) in &extended.matrix[global] {
            if pos[row] != usize::MAX && pos[row] != trace {
                let i = pos[row];
                entries.push((i, j, v));
                scale[i] = scale[i].max(v.abs());
            }
        }
        if global < g.n {
            entries.push((trace, j, 1.));
            scale[trace] = 1.;
        }
    }
    if scale.iter().any(|&v| v == 0.) {
        return Err(
            "Periodic steady space is nonunique; an initial-state projection is required".into(),
        );
    }
    let quick = block_stationary_solve(
            lap,
            &entries,
            r,
            trace,
            &inds
                .iter()
                .enumerate()
                .filter_map(|(i, &g)| (g < extended.physical_dimension).then_some(i))
                .collect::<Vec<_>>(),
            extended
                .frequency_basis
                .frequencies
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            require_condition,
            structured_stationary_products(model,&extended,&inds,trace).ok(),
            refinement_layout(&extended,&inds,&vec![1.;extended.matrix.len()])?,
        )
        .ok();
    let (linear, symbolic_cached) = if let Some(result) = quick {
        timings.insert("blockIterative".into(), json!(true));
        (result, false)
    } else {
        let triplets: Vec<_> = entries
            .iter()
            .map(|&(row, col, v)| Triplet::new(row, col, v / scale[row]))
            .collect();
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(r, r, &triplets)
            .map_err(|e| format!("Sparse construction: {e:?}"))?;
        timings.insert(
            "assembleAndScaleMs".into(),
            json!(phase.elapsed().as_secs_f64() * 1000.),
        );
        phase = Instant::now();
        let pattern = a.as_ref().symbolic();
        let cached_symbolic = STATIONARY_SYMBOLIC.with(|cache| {
            let mut cache = cache.borrow_mut();
            let i = cache.iter().position(|entry| {
                entry.dimension == r
                    && entry.column_pointers == pattern.col_ptr()
                    && entry.row_indices == pattern.row_idx()
            });
            i.map(|i| cache.remove(i).unwrap())
        });
        let symbolic_cached = cached_symbolic.is_some();
        let symbolic_entry = match cached_symbolic {
            Some(entry) => entry,
            None => StationarySymbolic {
                dimension: r,
                column_pointers: pattern.col_ptr().to_vec(),
                row_indices: pattern.row_idx().to_vec(),
                symbolic: std::sync::Arc::new(
                    faer::sparse::linalg::lu::factorize_symbolic_lu(pattern, Default::default())
                        .map_err(|e| format!("Periodic symbolic LU failed: {e:?}"))?,
                ),
            },
        };
        let symbolic = symbolic_entry.symbolic.clone();
        STATIONARY_SYMBOLIC.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.push_back(symbolic_entry);
            while cache.len() > 6 {
                cache.pop_front();
            }
        });
        timings.insert(
            "symbolicLuMs".into(),
            json!(phase.elapsed().as_secs_f64() * 1000.),
        );
        let mut column_sums = vec![0.; r];
        for &(i, j, v) in &entries {
            column_sums[j] += (v / scale[i]).abs();
        }
        let linear = stationary_linear_solve(
            &symbolic,
            a.as_ref(),
            trace,
            column_sums.into_iter().fold(0., f64::max),
        )?;
        (linear, symbolic_cached)
    };
    let condition = linear.condition;
    timings.insert("numericLuMs".into(), json!(linear.numeric_ms));
    timings.insert("conditionEstimateMs".into(), json!(linear.condition_ms));
    timings.insert("conditionDeferred".into(), json!(linear.pending_condition.is_some()));
    phase = Instant::now();
    let mut coefficients = vec![0.; extended.matrix.len()];
    for (i, &global) in inds.iter().enumerate() {
        coefficients[global] = linear.values[i];
    }
    let remainder = sparse_product(&extended.matrix, &coefficients);
    let residual = max_abs(&remainder);
    timings.insert(
        "solveAndResidualMs".into(),
        json!(linear.solve_ms + phase.elapsed().as_secs_f64() * 1000.),
    );
    phase = Instant::now();
    let mut trace_error = 0_f64;
    let mut positivity_error = 0_f64;
    for sample in 0..64 {
        let phases = phase_sample(extended.frequency_basis.frequencies.len(), sample);
        let x = reconstruct(&extended, &coefficients, &phases);
        trace_error = trace_error.max((x[..g.n].iter().sum::<f64>() - 1.).abs());
        positivity_error =
            positivity_error.max(positivity(lap, &expand_density(&g.coords, &x, g.n), g.n)?);
    }
    timings.insert(
        "densityChecksMs".into(),
        json!(phase.elapsed().as_secs_f64() * 1000.),
    );
    phase = Instant::now();
    let fourier = fourier_coefficients(&extended, &coefficients);
    let mut retained = vec![];
    for h in &extended.harmonics {
        retained.push(h.index.clone());
        if h.imaginary {
            retained.push(h.index.iter().map(|i| -i).collect());
        }
    }
    let drives: Vec<_> = model
        .drives
        .iter()
        .enumerate()
        .map(|(i, drive)| {
            Ok(FourierDrive {
                index: extended.frequency_basis.drive_indices[i].clone(),
                cos: hamiltonian_generator(model, &drive.cos)?,
                sin: hamiltonian_generator(model, &drive.sin)?,
            })
        })
        .collect::<Result<_, String>>()?;
    let boundary = harmonics::boundary_forcing(&drives, &fourier, &retained)?;
    timings.insert(
        "boundaryForcingMs".into(),
        json!(phase.elapsed().as_secs_f64() * 1000.),
    );
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    STATIONARY_PROFILE.with(|profile|{if let Some(records)=profile.borrow_mut().as_mut(){
        records.push(json!({"order":order,"dimension":r,"nnz":entries.len(),"timings":timings,"elapsedMs":elapsed_ms,"conditionEstimate":condition,"residual":residual,"proofCached":false,"symbolicCached":symbolic_cached,"numericStorageReused":linear.workspace_reused}));
    }});
    if linear.pending_condition.is_none() && residual < 1e-5 && trace_error < 1e-8 && positivity_error < 1e-8 {
        let saved = StationaryOperatorResult {
            matrix: extended.matrix.clone(),
            physical_dimension: g.n,
            indices: inds.clone(),
            frequencies: extended.frequency_basis.frequencies.clone(),
            harmonics: harmonic_key,
            coefficients: coefficients.clone(),
            residual,
            boundary_residual: boundary.bound.max_coordinate,
            trace_error,
            positivity_error,
            condition,
        };
        STATIONARY_OPERATORS.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.push_back(saved);
            while cache.len() > 12
                || cache
                    .iter()
                    .map(|entry| {
                        entry.matrix.iter().map(|col| col.len() * 16).sum::<usize>()
                            + entry.coefficients.len() * 8
                    })
                    .sum::<usize>()
                    > 128 * 1024 * 1024
            {
                if cache.len() <= 1 {
                    break;
                }
                cache.pop_front();
            }
        });
    }
    Ok(Stationary {
        extended,
        coefficients,
        residual,
        boundary_residual: boundary.bound.max_coordinate,
        trace_error,
        positivity_error,
        dimension: r,
        condition,
        valid: true,
        proof_cached: false,
        symbolic_cached,
        pending_condition: linear.pending_condition,
        prepared:linear.prepared,
    })
}

fn certify_stationary_candidate(lap: &Lapack, current: &mut Stationary, order: i32) -> Result<(), String> {
    let Some(prepared) = current.pending_condition.take() else { return Ok(()); };
    let started = Instant::now();
    let result = block_condition_estimate(lap, &prepared);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    STATIONARY_PROFILE.with(|profile| {
        if let Some(records) = profile.borrow_mut().as_mut() {
            if let Some(record) = records.iter_mut().rev().find(|record| record["order"].as_i64() == Some(order as i64)) {
                record["timings"]["conditionEstimateMs"] = json!(elapsed_ms);
                record["timings"]["conditionDeferred"] = json!(false);
                record["elapsedMs"] = json!(record["elapsedMs"].as_f64().unwrap_or(0.) + elapsed_ms);
                record["conditionEstimate"] = json!(result.as_ref().ok());
            }
        }
    });
    current.condition = result?;
    Ok(())
}

fn unresolved_stationary(extended: Extended) -> Stationary {
    let dimension=reachable(&extended.matrix,&extended.initial).len();
    let size = extended.matrix.len();
    Stationary {
        extended, coefficients: vec![0.; size], residual: 0., boundary_residual: 0.,
        trace_error: 0., positivity_error: 0., dimension, condition: f64::INFINITY,
        valid: false, proof_cached: false, symbolic_cached: false, pending_condition: None,prepared:None,
    }
}

fn stationary_linear_pass(current:&Stationary)->bool{
    current.trace_error<1e-8&&current.residual<1e-5
}
fn stationary_numerics_pass(current:&Stationary)->bool{
    stationary_linear_pass(current)&&current.positivity_error<1e-8
}
fn stationary_harmonics_pass(current:&Stationary,delta:f64,mean_delta:f64)->bool{
    delta.max(mean_delta)<1e-6&&current.boundary_residual<1e-7
}
fn stationary_requires_correction(current:&Stationary,delta:f64,mean_delta:f64)->bool{
    current.valid&&stationary_harmonics_pass(current,delta,mean_delta)&&!stationary_linear_pass(current)
}
fn stationary_raw_correction(lap:&Lapack,prepared:&crate::block_solve::Prepared,a:&Sparse,rhs:&[f64],x:&mut[f64])->Result<(f64,f64),String>{
    if a.len()!=x.len()||rhs.len()!=x.len(){return Err("Stationary residual correction dimensions differ".into());}
    // Recompute with the COMPLETE ORIGINAL unscaled operator. Multiplying a
    // balanced residual by row scales can conceal the error being corrected.
    let product=sparse_product(a,x);let residual:Vec<_>=rhs.iter().zip(product).map(|(b,value)|b-value).collect();
    let before=max_abs(&residual);let correction=prepared.solve(lap,&residual)?;
    if correction.len()!=x.len(){return Err("Stationary residual correction returned the wrong dimension".into());}
    for(value,delta)in x.iter_mut().zip(correction){*value+=delta;}
    if x.iter().any(|value|!value.is_finite()){return Err("Non-finite stationary residual correction".into());}
    let product=sparse_product(a,x);let after=rhs.iter().zip(product).map(|(b,value)|(b-value).abs()).fold(0.,f64::max);
    Ok((before,after))
}
fn refresh_stationary_checks(lap:&Lapack,model:&Model,g:&Generator,current:&mut Stationary)->Result<(),String>{
    current.residual=max_abs(&sparse_product(&current.extended.matrix,&current.coefficients));
    current.trace_error=0.;current.positivity_error=0.;
    for sample in 0..64{
        let phases=phase_sample(current.extended.frequency_basis.frequencies.len(),sample);
        let x=reconstruct(&current.extended,&current.coefficients,&phases);
        current.trace_error=current.trace_error.max((x[..g.n].iter().sum::<f64>()-1.).abs());
        current.positivity_error=current.positivity_error.max(positivity(lap,&expand_density(&g.coords,&x,g.n),g.n)?);
    }
    let fourier=fourier_coefficients(&current.extended,&current.coefficients);let mut retained=vec![];
    for h in &current.extended.harmonics{retained.push(h.index.clone());if h.imaginary{retained.push(h.index.iter().map(|i|-i).collect());}}
    let drives=model.drives.iter().enumerate().map(|(i,drive)|Ok(FourierDrive{
        index:current.extended.frequency_basis.drive_indices[i].clone(),cos:hamiltonian_generator(model,&drive.cos)?,sin:hamiltonian_generator(model,&drive.sin)?,
    })).collect::<Result<Vec<_>,String>>()?;
    current.boundary_residual=harmonics::boundary_forcing(&drives,&fourier,&retained)?.bound.max_coordinate;
    Ok(())
}
/// Harmonic convergence and algebraic solution accuracy are independent.
/// Once the former passes, repair the same normalized equation instead of
/// adding Fourier blocks to compensate for a row-scaled linear residual.
fn refine_stationary_candidate(lap:&Lapack,model:&Model,g:&Generator,current:&mut Stationary,order:i32)->Result<(),String>{
    let started=Instant::now();let mut records=vec![];
    let indices=reachable(&current.extended.matrix,&current.extended.initial);
    let n=indices.len();let mut positions=vec![usize::MAX;current.extended.matrix.len()];
    for(local,&global)in indices.iter().enumerate(){positions[global]=local;}
    let trace=indices.iter().position(|&global|global<g.n).ok_or("Stationary correction has no normalization row")?;
    let mut a:Sparse=vec![vec![];n];
    for(j,&global)in indices.iter().enumerate(){
        for &(row,value)in &current.extended.matrix[global]{let local=positions[row];if local!=usize::MAX&&local!=trace{a[j].push((local,value));}}
        if global<g.n{a[j].push((trace,1.));}
    }
    compress(&mut a);
    let prepared=if let Some(prepared)=&current.pending_condition{std::rc::Rc::clone(prepared)}else{
        let central:Vec<_>=indices.iter().enumerate().filter_map(|(i,&global)|(global<current.extended.physical_dimension).then_some(i)).collect();
        let omega=current.extended.frequency_basis.frequencies.iter().copied().fold(f64::INFINITY,f64::min);
        let prepared=crate::block_solve::Prepared::new(lap,&a,&central,0.02*omega)?;
        let(forward,transpose)=structured_stationary_products(model,&current.extended,&indices,trace)?;
        let prepared=prepared.with_raw_product(forward).with_raw_transpose(transpose);
        std::rc::Rc::new(prepared)
    };
    let mut rhs=vec![0.;n];rhs[trace]=1.;let mut values:Vec<_>=indices.iter().map(|&i|current.coefficients[i]).collect();
    current.pending_condition=Some(std::rc::Rc::clone(&prepared));current.prepared=Some(std::rc::Rc::clone(&prepared));current.condition=f64::NAN;current.proof_cached=false;
    for round in 0..2{
        if round==1&&prepared.is_harmonic_extension()&&!prepared.has_fresh_native(){
            // An unavailable fresh factor retries the original equation with
            // the existing factor.
            let _=prepared.refresh_native_for_refinement();
        }
        let(before,after)=stationary_raw_correction(lap,&prepared,&a,&rhs,&mut values)
            .map_err(|error|format!("Stationary K={order} same-order residual correction {} failed: {error}",round+1))?;
        for(&global,&value)in indices.iter().zip(&values){current.coefficients[global]=value;}
        refresh_stationary_checks(lap,model,g,current)?;
        records.push(json!({"round":round+1,"normalizedRawBefore":before,"normalizedRawAfter":after,"physicalRawResidual":current.residual,
            "traceError":current.trace_error,"positivityError":current.positivity_error,"boundaryForcingMax":current.boundary_residual,"freshNative":prepared.has_fresh_native()}));
        // A finite Fourier truncation need not be positive. Once raw/trace
        // accuracy passes, a PSD-only failure belongs to harmonic refinement,
        // not another linear correction or a new factorization.
        if stationary_linear_pass(current){break;}
    }
    STATIONARY_PROFILE.with(|profile|{if let Some(records_out)=profile.borrow_mut().as_mut(){if let Some(record)=records_out.iter_mut().rev().find(|record|record["order"].as_i64()==Some(order as i64)){
        let elapsed=started.elapsed().as_secs_f64()*1000.;record["rawResidualCorrections"]=json!(records);record["timings"]["rawResidualCorrectionMs"]=json!(elapsed);
        record["elapsedMs"]=json!(record["elapsedMs"].as_f64().unwrap_or(0.)+elapsed);record["residual"]=json!(current.residual);
    }}});
    if !stationary_linear_pass(current){return Err(format!("Stationary K={order} passed harmonic refinement but failed unchanged linear gates after two same-order residual corrections (raw {:e}, trace {:e}, PSD {:e})",current.residual,current.trace_error,current.positivity_error));}
    Ok(())
}

fn stationary_progress(
    lap: &Lapack, model: &Model, g: &Generator, options: &Options,
    current: &Stationary, previous: Option<&Stationary>, previous_mean: Option<&Vec<f64>>,
) -> Result<(Option<Vec<f64>>, f64, f64, bool), String> {
    let mean = if current.valid {
        Some(current.coefficients[..g.n * g.n].to_vec())
    } else if options.include_asymptotic {
        initial_mean(lap, model, g, &current.extended)?
    } else { None };
    if options.steady_only && mean.is_none() {
        return Err("An initial-state dependent long-time mean could not be resolved reliably; no arbitrary stationary mixture was selected".into());
    }
    let delta = if let Some(old) = previous {
        harmonics::uniform_difference(
            &fourier_coefficients(&old.extended, &old.coefficients),
            &fourier_coefficients(&current.extended, &current.coefficients),
        )?.max_coordinate
    } else { f64::INFINITY };
    let mean_delta = match (mean.as_ref(), previous_mean) {
        (Some(a), Some(b)) => a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0., f64::max),
        (None, None) => 0., _ => f64::INFINITY,
    };
    let converged=stationary_harmonics_pass(current,delta,mean_delta)&&stationary_numerics_pass(current);
    Ok((mean, delta, mean_delta, converged))
}

fn solve(lap: &Lapack, model: &Model, options: &Options) -> Result<Value, String> {
    let validation = build_generator(model)?;
    if positivity(
        lap,
        &expand_density(&validation.coords, &validation.initial, validation.n),
        validation.n,
    )? > 1e-10
    {
        return Err("Initial periodic density is not positive semidefinite".into());
    }
    for drive in &model.drives {
        if !drive.omega.is_finite() || drive.omega <= 0. {
            return Err("Harmonic angular frequencies must be positive and finite".into());
        }
        hamiltonian_generator(model, &drive.cos)?;
        hamiltonian_generator(model, &drive.sin)?;
    }
    let basis = harmonics::model_frequency_basis(model, 16)?;
    if !basis.exact || basis.rounded_relations != 0 {
        return Err("A frequency relation holds only after floating-point rounding; it cannot establish an infinite-time mean. Finite propagation retains the original independent frequencies.".into());
    }
    let mut result = solve_inner(lap, model, options, validation)?;
    result["diagnostics"]["frequencyCoordinateSource"]=json!(basis.source());
    if let Some(mapping)=&basis.source_to_basis{result["diagnostics"]["carrierSourceToBasis"]=json!(mapping);}
    Ok(result)
}
fn solve_inner(lap: &Lapack, model: &Model, options: &Options, g: Generator) -> Result<Value, String> {
    let started = Instant::now();
    let mut previous: Option<Stationary> = None;
    let mut previous_mean: Option<Vec<f64>> = None;
    for order in 1..=12 {
        let mut current = match stationary_candidate(lap, model, &g, order, false) {
            Ok(value) => value,
            Err(_) => unresolved_stationary(extend(model, &g, order)?),
        };
        let (mut mean, mut delta, mut mean_delta, mut mean_converged) = stationary_progress(
            lap, model, &g, options, &current, previous.as_ref(), previous_mean.as_ref(),
        )?;
        if stationary_requires_correction(&current,delta,mean_delta){
            refine_stationary_candidate(lap,model,&g,&mut current,order)?;
            (mean,delta,mean_delta,mean_converged)=stationary_progress(lap,model,&g,options,&current,previous.as_ref(),previous_mean.as_ref())?;
        }
        if mean_converged && current.pending_condition.is_some() {
            // Only the orders capable of contributing a returned mean pay for
            // the full inverse-norm estimate. If that check fails, retain the
            // original complete-LU/nonunique fallback.
            if certify_stationary_candidate(lap, &mut current, order).is_err() {
                current = match stationary(lap, model, &g, order) {
                    Ok(value) => value,
                    Err(_) => unresolved_stationary(extend(model, &g, order)?),
                };
                (mean, delta, mean_delta, _) = stationary_progress(
                    lap, model, &g, options, &current, previous.as_ref(), previous_mean.as_ref(),
                )?;
                if stationary_requires_correction(&current,delta,mean_delta){
                    refine_stationary_candidate(lap,model,&g,&mut current,order)?;
                    (mean,delta,mean_delta,mean_converged)=stationary_progress(lap,model,&g,options,&current,previous.as_ref(),previous_mean.as_ref())?;
                    if mean_converged&&current.pending_condition.is_some(){certify_stationary_candidate(lap,&mut current,order)?;}
                }
            }
        }
        // Harmonic density refinement targets 1e-6; residual, boundary
        // forcing, trace and positivity checks are independent gates.
        if delta.max(mean_delta) < 1e-6
            && current.trace_error < 1e-8
            && current.positivity_error < 1e-8
            && current.residual < 1e-5
            && current.boundary_residual < 1e-7
        {
            if current.valid && !current.condition.is_finite() {
                return Err("An uncertified stationary candidate cannot be returned".into());
            }
            let rho = mean.as_ref().map(|x| expand_density(&g.coords, x, g.n));
            let states: Vec<String> = if model.states.len() == g.n {
                model.states.clone()
            } else {
                (0..g.n).map(|i| i.to_string()).collect()
            };
            let asym = if options.include_asymptotic {
                mean.as_ref().map(|x| x[..g.n].to_vec())
            } else {
                None
            };
            let status = if !options.include_asymptotic {
                "not-requested"
            } else if current.valid || mean.is_some() {
                "initial-dependent-mean"
            } else {
                "unresolved-initial-dependent"
            };
            return Ok(
                json!({"timesUs":[0.],"states":states,"populations":[g.initial[..g.n].to_vec()],
                    "scatteringRatePerUs":[scatter_rate(&g.scattering,&expand_density(&g.coords,&g.initial,g.n),g.n)],
                    "asymptoticFourierDensity":if options.include_asymptotic&&current.valid{Some(fourier_density(&g,&current.extended,&current.coefficients))}else{None},"asymptoticDensityReal":rho.as_ref().map(|rho|rho.iter().map(|z|z.re).collect::<Vec<_>>()),"asymptoticDensityImag":rho.as_ref().map(|rho|rho.iter().map(|z|z.im).collect::<Vec<_>>()),"asymptoticPopulations":asym,"asymptoticScatteringRatePerUs":rho.as_ref().map(|rho|scatter_rate(&g.scattering,rho,g.n)),
                    "asymptoticStatus":status,"diagnostics":{"method":"Rust / faer full-density Fourier-Galerkin Lindblad solve",
                        "engine":"rust-floquet","liouvilleDimension":g.n*g.n,"reachableDimension":reachable(&current.extended.matrix,&current.extended.initial).len(),"harmonicOrder":order,
                        "stationaryOperatorDimension":if current.valid{Some(current.dimension)}else{None},
                        "stationaryProofCached":current.proof_cached,"stationarySymbolicCached":current.symbolic_cached,
                        "uniquenessScope":"phase-average solution on the exact invariant coordinate space of the specified initial density; no global uniqueness asserted",
                        "harmonicConvergenceError":delta.max(mean_delta),"independentBeats":current.extended.frequency_basis.frequencies.len(),"stationaryOnly":options.steady_only,"initialDependent":true,
                        "boundaryForcingMax":current.boundary_residual,"frequencyBasis":current.extended.frequency_basis.frequencies,
                        "stationarySystemConditionEstimate":if current.valid{Some(current.condition)}else{None},
                        "frequencyReconstructionErrors":current.extended.frequency_basis.reconstruction_errors,"roundedFrequencyRelations":current.extended.frequency_basis.rounded_relations,
                        "elapsedMs":started.elapsed().as_secs_f64()*1000.,"traceError":current.trace_error,"positivityError":current.positivity_error,
                        "hermiticityError":0.,"residual":current.residual,"asymptoticStatus":status,"densityLimitExists":false,
                        "asymptoticReason":"Converged harmonic solution of the full driven Lindblad equation; populations are the long-time phase average, not a time-independent density."}}
                ),
            );
        }
        previous = Some(current);
        previous_mean = mean;
    }
    Err("Fourier harmonic refinement did not converge within order 12; no stationary mean is asserted".into())
}
