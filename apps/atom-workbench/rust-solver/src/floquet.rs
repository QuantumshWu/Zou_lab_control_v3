//! Fourier-Galerkin solution of the full time-dependent Lindblad equation.
//! No electronic state or density coherence is eliminated. Harmonic order is
//! increased and the reconstructed physical density is checked for convergence.
use super::*;
use crate::harmonics::{FourierCoefficient, FourierDrive, FrequencyBasis};
use faer::{
    linalg::solvers::Solve,
    sparse::{SparseColMat, Triplet},
    Mat,
};
use std::cell::{Cell, RefCell};
thread_local! {static FULL_DRIVE_MODE:Cell<bool>=const{Cell::new(false)};}
thread_local! {static SHARED_SCHUR:RefCell<Option<std::rc::Rc<crate::schur_shift::Shared>>>=const{RefCell::new(None)};}
struct RecycledSpace {
    indices: Vec<usize>,
    columns: Vec<Vec<f64>>,
}
thread_local! {static RECYCLED_SPACE:RefCell<Option<RecycledSpace>>=const{RefCell::new(None)};}
#[derive(Clone,Debug,Eq,PartialEq,Hash)]
struct RefinementCoordinate{harmonic:Vec<i32>,part:usize,physical:usize}
#[derive(Clone,PartialEq)]
struct RefinementLayout{coordinates:Vec<RefinementCoordinate>,weights:Vec<f64>,frequencies:Vec<f64>,drive_indices:Vec<Vec<i32>>,physical_dimension:usize}
struct ReusableResolvent{
    identity:std::rc::Rc<()>,layout:RefinementLayout,shift:f64,frequency:f64,
    matrix:Sparse,factor:std::rc::Rc<crate::block_solve::RawPreconditioner>,
}
thread_local! {static REFINEMENT_PRECONDITIONER:RefCell<Option<ReusableResolvent>>=const{RefCell::new(None)};}
struct ReusableStationary{
    layout:RefinementLayout,trace:RefinementCoordinate,matrix:Sparse,
    prepared:std::rc::Rc<crate::block_solve::Prepared>,
}
// Separate from shifted trajectory factors: the stationary trace-row equation
// has a different operator and may never be reused across a numerical shift.
thread_local! {static MEAN_REFINEMENT_PRECONDITIONER:RefCell<Option<ReusableStationary>>=const{RefCell::new(None)};}
#[derive(Default,serde::Serialize)]
#[serde(rename_all="camelCase")]
struct ZeroPoleReport{attempts:usize,reused_orders:usize,extended_orders:usize,fallbacks:usize,setup_ms:f64,reasons:Vec<String>}
thread_local! {
    static ZERO_POLE_PREPARED:RefCell<Option<std::rc::Rc<ZeroGaugePrepared>>>=const{RefCell::new(None)};
    static ZERO_POLE_DISABLED:Cell<bool>=const{Cell::new(false)};
    static ZERO_POLE_REPORT:RefCell<ZeroPoleReport>=RefCell::new(ZeroPoleReport::default());
}
fn reset_zero_pole(){ZERO_POLE_PREPARED.with(|cache|*cache.borrow_mut()=None);ZERO_POLE_DISABLED.with(|flag|flag.set(false));ZERO_POLE_REPORT.with(|report|*report.borrow_mut()=ZeroPoleReport::default());}
fn full_drive_mode() -> bool {
    FULL_DRIVE_MODE.with(Cell::get)
}
pub(super) fn clear_request_results() {
    reset_zero_pole();
    REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
    MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
    STEADY_RESULTS.with(|v| v.borrow_mut().clear());
    STATIONARY_OPERATORS.with(|v| v.borrow_mut().clear());
    PROPAGATORS.with(|v| v.borrow_mut().clear());
    STATIONARY_SYMBOLIC.with(|v| v.borrow_mut().clear());
    STATIONARY_NUMERIC.with(|v| *v.borrow_mut() = None);
}
pub(super) fn solve_full(lap: &Lapack, model: &Model, options: &Options) -> Result<Value, String> {
    FULL_DRIVE_MODE.with(|flag| {
        let old = flag.replace(true);
        reset_zero_pole();
        SHARED_SCHUR.with(|s|*s.borrow_mut()=None);
        RECYCLED_SPACE.with(|cache| *cache.borrow_mut() = None);
        REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
        MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
        STATIONARY_PROFILE.with(|p| *p.borrow_mut() = Some(Vec::new()));
        let result = solve(lap, model, options).map(|mut result| {
            result["diagnostics"]["stationaryPhases"] =
                STATIONARY_PROFILE.with(|p| json!(p.borrow_mut().take()));
            ZERO_POLE_REPORT.with(|report|{if report.borrow().attempts>0{result["diagnostics"]["zeroPoleMeanReuse"]=json!(&*report.borrow());}});
            SHARED_SCHUR.with(|s|{if let Some(s)=s.borrow().as_ref(){result["diagnostics"]["schurExperiment"]=json!({"setupMs":s.setup_ms,"preparations":s.preparations.get(),"applications":s.applications.get()});}});
            result
        });
        RECYCLED_SPACE.with(|cache| *cache.borrow_mut() = None);
        REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
        MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=None);
        ZERO_POLE_PREPARED.with(|cache|*cache.borrow_mut()=None);
        SHARED_SCHUR.with(|s|*s.borrow_mut()=None);
        flag.set(old);
        result
    })
}
#[cfg(test)]
mod full_drive_regression {
    use super::*;
    #[test]
    fn strong_two_carrier_long_trajectory_waits_for_harmonic_convergence(){
        let request:Request=serde_json::from_str(include_str!("../tests/fixtures/strong-two-carrier.json")).unwrap();
        let model=request.model.unwrap();let lap=Lapack::load().unwrap();clear_request_results();
        let result=solve_full(&lap,&model,&request.options).unwrap();
        assert_eq!(result["timesUs"].as_array().unwrap().len(),160);
        assert_eq!(result["timesUs"][159],100000.);
        assert!(result["diagnostics"]["harmonicOrder"].as_i64().unwrap()>=4);
        assert!(result["diagnostics"]["krylovDimension"].as_u64().unwrap()<128);
        assert!(result["diagnostics"]["positivityError"].as_f64().unwrap()<1e-6);
        // A separately rebuilt, higher Fourier order checks the complete
        // returned trajectory, including the weak long-time population modes.
        FULL_DRIVE_MODE.with(|f|f.set(true));RECYCLED_SPACE.with(|s|*s.borrow_mut()=None);
        let g=build_generator(&model).unwrap();let fifth=stationary(&lap,&model,&g,5).unwrap();
        let reference=trajectory(&lap,&model,&g,&fifth,&request.options);
        FULL_DRIVE_MODE.with(|f|f.set(false));RECYCLED_SPACE.with(|s|*s.borrow_mut()=None);
        let reference=reference.unwrap();let mut difference=0_f64;
        for(a,b)in result["populations"].as_array().unwrap().iter().zip(reference.states){for(j,v)in a.as_array().unwrap().iter().enumerate(){difference=difference.max((v.as_f64().unwrap()-b[j]).abs());}}
        assert!(difference<1e-5,"Higher-order trajectory difference {difference:e}");
    }
    #[test]
    fn unstable_recycled_projection_is_rejected_before_large_basis_growth(){
        let request:Request=serde_json::from_str(include_str!("../tests/fixtures/strong-two-carrier.json")).unwrap();
        let model=request.model.unwrap();let lap=Lapack::load().unwrap();let g=build_generator(&model).unwrap();
        clear_request_results();FULL_DRIVE_MODE.with(|f|f.set(true));RECYCLED_SPACE.with(|s|*s.borrow_mut()=None);
        let first=stationary(&lap,&model,&g,1).unwrap();trajectory(&lap,&model,&g,&first,&request.options).unwrap();
        let second=stationary(&lap,&model,&g,2).unwrap();let result=trajectory(&lap,&model,&g,&second,&request.options);
        FULL_DRIVE_MODE.with(|f|f.set(false));RECYCLED_SPACE.with(|s|*s.borrow_mut()=None);
        let reason=result.err().expect("The deliberately recycled low-order space must be rejected");
        assert!(reason.contains("Unstable projected exponential"),"{reason}");
        assert!(!reason.contains("768"),"Do not spend the entire basis budget on nonphysical growth");
    }
    #[test]
    fn full_three_beam_matches_independent_dense_fourier_population() {
        let request: Request =
            serde_json::from_str(include_str!("../tests/fixtures/full-three-beam.json")).unwrap();
        let model = request.model.unwrap();
        let lap = Lapack::load().unwrap();
        let result = solve_full(&lap, &model, &request.options).unwrap();
        let oracle: Value =
            serde_json::from_str(include_str!("../tests/fixtures/full-three-oracle.json")).unwrap();
        let mut error = 0_f64;
        for (a, b) in result["populations"]
            .as_array()
            .unwrap()
            .iter()
            .zip(oracle["populations"].as_array().unwrap())
        {
            for (a, b) in a.as_array().unwrap().iter().zip(b.as_array().unwrap()) {
                error = error.max((a.as_f64().unwrap() - b.as_f64().unwrap()).abs());
            }
        }
        assert!(error < 1e-4, "Full-model population error {error:e}");
        assert!(
            result["asymptoticFourierDensity"]["harmonics"]
                .as_array()
                .unwrap()
                .len()
                >= 2
        );
        let strict = solve(
            &lap,
            &model,
            &Options {
                steady_only: true,
                ..Options::default()
            },
        )
        .unwrap();
        for (a, b) in result["asymptoticPopulations"]
            .as_array()
            .unwrap()
            .iter()
            .zip(strict["asymptoticPopulations"].as_array().unwrap())
        {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-7);
        }
    }
}

use std::collections::{BTreeSet, HashMap};

type Sparse = Vec<Vec<(usize, f64)>>;
struct PropagatorCache {
    matrix: Sparse,
    prepared: crate::cluster_action::Prepared,
}
thread_local! {static PROPAGATORS:RefCell<VecDeque<PropagatorCache>>=const{RefCell::new(VecDeque::new())};}
thread_local! {static STEADY_RESULTS:RefCell<VecDeque<(Model,Value)>>=const{RefCell::new(VecDeque::new())};}
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
fn same_model(a: &Model, b: &Model) -> bool {
    let matrix = |a: &Operator, b: &Operator| a.real == b.real && a.imag == b.imag;
    a.dimension == b.dimension
        && a.states == b.states
        && a.h_real == b.h_real
        && a.h_imag == b.h_imag
        && a.initial_index == b.initial_index
        && a.initial_real == b.initial_real
        && a.initial_imag == b.initial_imag
        && a.frequency_coordinates == b.frequency_coordinates
        && a.collapse.len() == b.collapse.len()
        && a.collapse
            .iter()
            .zip(&b.collapse)
            .all(|(a, b)| matrix(a, b))
        && a.drives.len() == b.drives.len()
        && a.drives
            .iter()
            .zip(&b.drives)
            .all(|(a, b)| a.omega == b.omega && matrix(&a.cos, &b.cos) && matrix(&a.sin, &b.sin))
}
fn refined_action(
    lap: &Lapack,
    a: &Sparse,
    initial: &[f64],
    times: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let cached = PROPAGATORS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let index = cache.iter().position(|entry| entry.matrix == *a);
        index.map(|i| cache.remove(i).unwrap())
    });
    let entry = match cached {
        Some(entry) => entry,
        None => PropagatorCache {
            matrix: a.clone(),
            prepared: crate::cluster_action::prepare(lap, a, 1e5)?,
        },
    };
    let result = entry.prepared.apply(lap, initial, times)?;
    if result.initial_error > 1e-8 {
        return Err("Refined Fourier propagator failed initial-density reconstruction".into());
    }
    PROPAGATORS.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.push_back(entry);
        while cache.len() > 3
            || cache
                .iter()
                .map(|e| e.matrix.len() * e.matrix.len() * 16)
                .sum::<usize>()
                > 512 * 1024 * 1024
        {
            if cache.len() <= 1 {
                break;
            }
            cache.pop_front();
        }
    });
    Ok(result.states)
}
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
fn refinement_embedding(old:&RefinementLayout,new:&RefinementLayout,old_shift:f64,new_shift:f64,old_frequency:f64,new_frequency:f64)->Result<Vec<usize>,String>{
    if old_shift!=new_shift||old_frequency!=new_frequency||old.physical_dimension!=new.physical_dimension
        ||old.frequencies!=new.frequencies||old.drive_indices!=new.drive_indices{
        return Err("Refinement preconditioner frequency basis, physical coordinates, shift or pole changed".into());
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
    fn trajectory_candidate_rejects_coherence_error_even_when_populations_match() {
        let lap=Lapack::load().unwrap();
        let g=build_generator(&rotating_two_level(7.)).unwrap();
        let state=|coherence:f64| g.coords.iter().map(|coord|match coord {
            Coordinate::Population(_)=>0.5,
            Coordinate::Real(_,_)=>coherence,
            Coordinate::Imag(_,_)=>0.,
        }).collect::<Vec<_>>();
        let valid=state(0.49999);let invalid=state(0.50013);
        assert_eq!(valid[..2],invalid[..2]);
        let good=trajectory_density_error(&lap,&g,&[valid.clone()]).unwrap();
        let bad=trajectory_density_error(&lap,&g,&[valid,invalid]).unwrap();
        assert!(good.0<1e-12&&good.1<1e-12);
        assert!(bad.0<1e-12&&bad.1>1.29e-4);
    }

    #[test]
    fn deferred_stationary_condition_keeps_the_same_final_gate_and_coefficients() {
        let lap = Lapack::load().unwrap();
        let model = rotating_two_level(7.);
        let g = build_generator(&model).unwrap();
        clear_request_results();
        FULL_DRIVE_MODE.with(|flag| flag.set(true));
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
        FULL_DRIVE_MODE.with(|flag| flag.set(false));
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
            assert_eq!(out["asymptoticStatus"], "periodic-mean");
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
    fn rotating_drive_transient_matches_constant_frame_full_density() {
        let lap = Lapack::load().unwrap();
        let m = rotating_two_level(17.);
        let options = Options {
            duration_us: 3.,
            samples: 41,
            ..Options::default()
        };
        let actual = solve(&lap, &m, &options).unwrap();
        let mut constant = m.clone();
        constant.drives.clear();
        constant.time_dependent = false;
        constant.h_real = vec![0., 0.7, 0.7, 1.3];
        let expected = Solver::new().unwrap().solve(&constant, &options).unwrap();
        for t in 0..41 {
            for i in 0..2 {
                assert!(
                    (actual["populations"][t][i].as_f64().unwrap()
                        - expected["populations"][t][i].as_f64().unwrap())
                    .abs()
                        < 1e-10
                );
            }
        }
    }
    #[test]
    #[ignore = "Reads the generated Rb87 model fixture for a full-system performance and convergence audit"]
    fn default_rb32_transient_audit() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/v4/default-master.json"
        ))
        .unwrap();
        let model: Model = serde_json::from_str(&source).unwrap();
        let lap = Lapack::load().unwrap();
        let result = solve(
            &lap,
            &model,
            &Options {
                duration_us: 20.,
                samples: 161,
                ..Options::default()
            },
        );
        match result {
            Ok(value) => println!("{}", value["diagnostics"]),
            Err(error) => panic!("{error}"),
        }
    }
    #[test]
    #[ignore = "Dense full-space reference for testing the accelerated exponential"]
    fn default_rb32_dense_benchmark() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/v4/default-master.json"
        ))
        .unwrap();
        let model: Model = serde_json::from_str(&source).unwrap();
        let lap = Lapack::load().unwrap();
        let g = build_generator(&model).unwrap();
        let order = std::env::var("ATOM_FLOQUET_ORDER")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(1);
        let st = stationary(&lap, &model, &g, order).unwrap();
        let started = Instant::now();
        let tr = trajectory(
            &lap,
            &model,
            &g,
            &st,
            &Options {
                duration_us: 20.,
                samples: 161,
                ..Options::default()
            },
        )
        .unwrap();
        let path = format!(
            "{}/../artifacts/v4/native-dense-k{order}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::write(path,json!({"harmonicOrder":order,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"timesUs":tr.times,"coordinates":tr.states}).to_string()).unwrap();
        println!(
            "Dense K={order}, elapsed={}ms",
            started.elapsed().as_secs_f64() * 1000.
        );
    }
    #[test]
    fn independent_full_density_fixtures() {
        let fixtures: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/floquet-independent-cases.json"
        ))
        .unwrap();
        let lap = Lapack::load().unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let mut options: Options = serde_json::from_value(case["options"].clone()).unwrap();
            options.return_density = true;
            let actual = solve(&lap, &model, &options)
                .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
            let mut error = 0_f64;
            for key in ["densityReal", "densityImag"] {
                for (row, expected) in actual[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(case["reference"][key].as_array().unwrap())
                {
                    for (a, b) in row
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(expected.as_array().unwrap())
                    {
                        error = error.max((a.as_f64().unwrap() - b.as_f64().unwrap()).abs());
                    }
                }
            }
            assert!(
                error < 2e-8,
                "{}: full density error {error:e}",
                case["name"]
            );
            println!(
                "{}: density error={error:e}, harmonics={}",
                case["name"], actual["diagnostics"]["harmonicOrder"]
            );
            if let Some(expected) = case["reference"]["asymptoticPopulations"].as_array() {
                for (a, b) in actual["asymptoticPopulations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(expected)
                {
                    assert!(
                        (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 2e-8,
                        "{} long-time mean",
                        case["name"]
                    );
                }
            }
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
    fn stationary_uniqueness_uses_complete_physical_operator_algebra() {
        let lap = Lapack::load().unwrap();
        // A periodic identity Hamiltonian has no physical effect. With a
        // ground-state initial density, constant Rabi dynamics reach only
        // {p0,p1,Im rho01}; their trace-LU is invertible. The missing Re rho01
        // is nevertheless a conserved physical observable, so uniqueness
        // cannot be certified on that initial-coordinate graph.
        let model = Model { frequency_coordinates:None,
            dimension: 2,
            states: vec![],
            h_real: vec![0., 0.5, 0.5, 0.],
            h_imag: vec![0.; 4],
            collapse: vec![],
            initial_index: 0,
            initial_real: None,
            initial_imag: None,
            time_dependent: true,
            drives: vec![Drive {
                omega: 1.3,
                cos: Operator {
                    real: vec![1., 0., 0., 1.],
                    imag: vec![0.; 4],
                },
                sin: Operator {
                    real: vec![0.; 4],
                    imag: vec![0.; 4],
                },
            }],
        };
        let g = build_generator(&model).unwrap();
        let ext = extend(&model, &g, 1).unwrap();
        assert_eq!(reachable(&ext.matrix, &ext.initial).len(), 3);
        assert_eq!(
            physical_operator_indices(&model, &g, &ext).unwrap().len(),
            12
        );
        assert!(stationary(&lap, &model, &g, 1).is_err());
        let output = solve(
            &lap,
            &model,
            &Options {
                steady_only: true,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(output["asymptoticStatus"], "initial-dependent-mean");
        assert!((output["asymptoticPopulations"][0].as_f64().unwrap() - 0.5).abs() < 1e-12);
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
        let finite = solve(
            &lap,
            &model,
            &Options {
                duration_us: 0.2,
                samples: 5,
                return_density: true,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(finite["asymptoticStatus"], "not-converged");
        assert!(finite["asymptoticPopulations"].is_null());
        assert_eq!(finite["timesUs"].as_array().unwrap().len(), 5);
        assert!(
            finite["diagnostics"]["unmergedRoundedFrequencyRelations"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
    #[test]
    fn full_operator_proof_restricts_only_exact_physical_hilbert_support() {
        let mut model = rotating_two_level(3.);
        model.dimension = 3;
        model.states.clear();
        let pad = |entries: &[f64]| {
            vec![
                entries[0], entries[1], 0., entries[2], entries[3], 0., 0., 0., 0.,
            ]
        };
        model.h_real = pad(&model.h_real);
        model.h_imag = pad(&model.h_imag);
        for op in &mut model.collapse {
            op.real = pad(&op.real);
            op.imag = pad(&op.imag);
        }
        for drive in &mut model.drives {
            for op in [&mut drive.cos, &mut drive.sin] {
                op.real = pad(&op.real);
                op.imag = pad(&op.imag);
            }
        }
        let g = build_generator(&model).unwrap();
        let ext = extend(&model, &g, 2).unwrap();
        let indices = physical_operator_indices(&model, &g, &ext).unwrap();
        assert_eq!(indices.len(), 20);
        let result = solve(
            &Lapack::load().unwrap(),
            &model,
            &Options {
                steady_only: true,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(result["asymptoticStatus"], "periodic-mean");
        assert_eq!(result["asymptoticPopulations"][2], 0.);
    }
    #[test]
    fn stationary_operator_cache_keeps_current_initial_and_jump_observable() {
        let lap = Lapack::load().unwrap();
        let model = rotating_two_level(4.71);
        let options = Options {
            steady_only: true,
            ..Options::default()
        };
        let first = solve(&lap, &model, &options).unwrap();
        let mut changed = model.clone();
        changed.initial_index = 1;
        changed.states = vec!["ground".into(), "excited".into()];
        changed.collapse.push(Operator {
            real: vec![1., 0., 0., 1.],
            imag: vec![0.; 4],
        });
        let second = solve(&lap, &changed, &options).unwrap();
        assert_eq!(second["diagnostics"]["stationaryProofCached"], true);
        assert_eq!(second["populations"], json!([[0., 1.]]));
        assert_eq!(second["states"], json!(["ground", "excited"]));
        assert!((second["scatteringRatePerUs"][0].as_f64().unwrap() - 2.).abs() < 1e-12);
        assert!(
            (second["asymptoticScatteringRatePerUs"].as_f64().unwrap()
                - first["asymptoticScatteringRatePerUs"].as_f64().unwrap()
                - 1.)
                .abs()
                < 1e-12
        );
        assert_eq!(
            first["asymptoticPopulations"],
            second["asymptoticPopulations"]
        );
        changed.h_real[3] += 0.03;
        let third = solve(&lap, &changed, &options).unwrap();
        assert_eq!(third["diagnostics"]["stationaryProofCached"], false);
        assert_eq!(third["diagnostics"]["stationarySymbolicCached"], true);
        assert!(
            (third["asymptoticPopulations"][1].as_f64().unwrap()
                - second["asymptoticPopulations"][1].as_f64().unwrap())
            .abs()
                > 1e-6
        );
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
    #[test]
    #[ignore = "Full Rb87 stationary performance/convergence audit"]
    fn default_rb32_stationary_audit() {
        initialize_numeric_parallelism();
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/v4/default-master.json"
        ))
        .unwrap();
        let model: Model = serde_json::from_str(&source).unwrap();
        let result = solve(
            &Lapack::load().unwrap(),
            &model,
            &Options {
                steady_only: true,
                ..Options::default()
            },
        )
        .unwrap();
        println!("{}", result["diagnostics"]);
        let f1 = (0..3)
            .map(|i| result["asymptoticPopulations"][i].as_f64().unwrap())
            .sum::<f64>();
        assert!((f1 - 0.9993743141298147).abs() < 1e-8);
    }
    #[test]
    #[ignore = "Phase profile of full-Hilbert stationary solves under physical parameter changes"]
    fn stationary_phase_profile_audit() {
        initialize_numeric_parallelism();
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/stationary-performance");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let lap = Lapack::load().unwrap();
        let mut reports = Vec::new();
        for case in cases.as_array().unwrap() {
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let before = Instant::now();
            let g = build_generator(&model).unwrap();
            let generator_ms = before.elapsed().as_secs_f64() * 1000.;
            STATIONARY_PROFILE.with(|profile| *profile.borrow_mut() = Some(Vec::new()));
            for order in 1..=3 {
                let result = stationary(&lap, &model, &g, order).unwrap();
                assert!(
                    result.residual < 1e-5
                        && result.trace_error < 1e-8
                        && result.positivity_error < 1e-8
                );
            }
            let profile = STATIONARY_PROFILE.with(|profile| profile.borrow_mut().take().unwrap());
            let report = json!({"name":case["name"],"generatorMs":generator_ms,"orders":profile});
            println!("{report}");
            reports.push(report);
        }
        std::fs::write(
            folder.join("phase-profile.json"),
            serde_json::to_string_pretty(&reports).unwrap(),
        )
        .unwrap();
    }
    #[test]
    #[ignore = "Exact disconnected-component stationary factorization benchmark; every zero-RHS block is still certified"]
    fn stationary_exact_components_audit() {
        initialize_numeric_parallelism();
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/stationary-performance");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let lap = Lapack::load().unwrap();
        let mut records = Vec::new();
        for case in cases.as_array().unwrap().iter().filter(|case| {
            matches!(
                case["name"].as_str(),
                Some("default-cold" | "B-magnitude-change")
            )
        }) {
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let g = build_generator(&model).unwrap();
            for order in [1, 2, 3, 2, 3] {
                STATIONARY_OPERATORS.with(|cache| cache.borrow_mut().clear());
                let before = Instant::now();
                let reference = stationary(&lap, &model, &g, order).unwrap();
                let full_ms = before.elapsed().as_secs_f64() * 1000.;
                let ext = extend(&model, &g, order).unwrap();
                let indices = physical_operator_indices(&model, &g, &ext).unwrap();
                let n = indices.len();
                let mut positions = vec![usize::MAX; ext.matrix.len()];
                for (i, &global) in indices.iter().enumerate() {
                    positions[global] = i;
                }
                let trace = indices.iter().position(|&i| i < g.n).unwrap();
                let started = Instant::now();
                let mut columns = vec![Vec::new(); n];
                let mut scales = vec![0_f64; n];
                let mut edges = vec![Vec::new(); n];
                for (j, &global) in indices.iter().enumerate() {
                    for &(row, value) in &ext.matrix[global] {
                        let i = positions[row];
                        if i != trace {
                            columns[j].push((i, value));
                            scales[i] = scales[i].max(value.abs());
                        }
                    }
                    if global < g.n {
                        columns[j].push((trace, 1.));
                        scales[trace] = 1.;
                    }
                }
                for (j, entries) in columns.iter_mut().enumerate() {
                    for (i, value) in entries {
                        *value /= scales[*i];
                        if *value != 0. && *i != j {
                            edges[*i].push(j);
                            edges[j].push(*i);
                        }
                    }
                }
                let mut groups = Vec::new();
                let mut seen = vec![false; n];
                for start in 0..n {
                    if seen[start] {
                        continue;
                    }
                    let mut todo = vec![start];
                    seen[start] = true;
                    let mut group = Vec::new();
                    while let Some(i) = todo.pop() {
                        group.push(i);
                        for &j in &edges[i] {
                            if !seen[j] {
                                seen[j] = true;
                                todo.push(j);
                            }
                        }
                    }
                    group.sort_unstable();
                    groups.push(group);
                }
                let graph_ms = started.elapsed().as_secs_f64() * 1000.;
                let mut output = vec![0.; ext.matrix.len()];
                let mut block_reports = Vec::new();
                let mut operator_norm = 0_f64;
                let mut inverse_norm = 0_f64;
                for group in groups {
                    let size = group.len();
                    let mut local = vec![usize::MAX; n];
                    for (i, &global) in group.iter().enumerate() {
                        local[global] = i;
                    }
                    let before = Instant::now();
                    let mut triplets = Vec::new();
                    let mut one = 0_f64;
                    for (j, &global) in group.iter().enumerate() {
                        let mut sum = 0.;
                        for &(row, value) in &columns[global] {
                            assert_ne!(local[row], usize::MAX);
                            triplets.push(Triplet::new(local[row], j, value));
                            sum += value.abs();
                        }
                        one = one.max(sum);
                    }
                    let a =
                        SparseColMat::<usize, f64>::try_new_from_triplets(size, size, &triplets)
                            .unwrap();
                    let assembly_ms = before.elapsed().as_secs_f64() * 1000.;
                    let before = Instant::now();
                    let symbolic =
                        faer::sparse::linalg::solvers::SymbolicLu::try_new(a.as_ref().symbolic())
                            .unwrap();
                    let symbolic_ms = before.elapsed().as_secs_f64() * 1000.;
                    let before = Instant::now();
                    let lu = faer::sparse::linalg::solvers::Lu::try_new_with_symbolic(
                        symbolic,
                        a.as_ref(),
                    )
                    .unwrap();
                    let numeric_ms = before.elapsed().as_secs_f64() * 1000.;
                    let before = Instant::now();
                    let inverse = inverse_one_norm(&lu, size);
                    assert!(inverse.is_finite());
                    operator_norm = operator_norm.max(one);
                    inverse_norm = inverse_norm.max(inverse);
                    let condition_ms = before.elapsed().as_secs_f64() * 1000.;
                    let has_rhs = local[trace] != usize::MAX;
                    let before = Instant::now();
                    if has_rhs {
                        let rhs =
                            Mat::<f64>::from_fn(
                                size,
                                1,
                                |i, _| if group[i] == trace { 1. } else { 0. },
                            );
                        let solution = lu.solve(&rhs);
                        for (i, &global) in group.iter().enumerate() {
                            output[indices[global]] = solution[(i, 0)];
                        }
                    }
                    block_reports.push(json!({"dimension":size,"hasRhs":has_rhs,"assemblyMs":assembly_ms,"symbolicMs":symbolic_ms,"numericMs":numeric_ms,"conditionMs":condition_ms,"solveMs":before.elapsed().as_secs_f64()*1000.,"inverseOneNormEstimate":inverse,"operatorOneNorm":one}));
                }
                let total_ms = started.elapsed().as_secs_f64() * 1000.;
                let delta = output
                    .iter()
                    .zip(&reference.coefficients)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f64, f64::max);
                let residual = max_abs(&sparse_product(&ext.matrix, &output));
                let condition = operator_norm * inverse_norm;
                assert!(condition < 1e12 && delta < 1e-8 && residual < 1e-5);
                let record = json!({"name":case["name"],"order":order,"fullSolveMs":full_ms,"componentSolveMs":total_ms,"graphMs":graph_ms,"blocks":block_reports,"conditionEstimate":condition,"maxCoefficientDifference":delta,"residual":residual});
                println!("{record}");
                records.push(record);
            }
        }
        std::fs::write(
            folder.join("component-profile.json"),
            serde_json::to_string_pretty(&records).unwrap(),
        )
        .unwrap();
    }
    #[test]
    #[ignore = "Compare algebraically equivalent central population trace rows without changing production selection"]
    fn stationary_trace_row_profile_audit() {
        initialize_numeric_parallelism();
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../artifacts/v4/stationary-performance");
        let cases: Value =
            serde_json::from_str(&std::fs::read_to_string(folder.join("cases.json")).unwrap())
                .unwrap();
        let lap = Lapack::load().unwrap();
        let mut records = Vec::new();
        for case in cases.as_array().unwrap().iter().filter(|case| {
            matches!(
                case["name"].as_str(),
                Some("default-cold" | "B-magnitude-change")
            )
        }) {
            let model: Model = serde_json::from_value(case["model"].clone()).unwrap();
            let g = build_generator(&model).unwrap();
            let reference = stationary(&lap, &model, &g, 3).unwrap();
            let ext = extend(&model, &g, 3).unwrap();
            let indices = physical_operator_indices(&model, &g, &ext).unwrap();
            let n = indices.len();
            let mut positions = vec![usize::MAX; ext.matrix.len()];
            for (i, &global) in indices.iter().enumerate() {
                positions[global] = i;
            }
            for population in [0, 3, 7, 8, 12, 16, 23, 31] {
                let trace = positions[population];
                let started = Instant::now();
                let mut entries = Vec::new();
                let mut scales = vec![0_f64; n];
                for (j, &global) in indices.iter().enumerate() {
                    for &(row, value) in &ext.matrix[global] {
                        let i = positions[row];
                        if i != trace {
                            entries.push((i, j, value));
                            scales[i] = scales[i].max(value.abs());
                        }
                    }
                    if global < g.n {
                        entries.push((trace, j, 1.));
                        scales[trace] = 1.;
                    }
                }
                let triplets: Vec<_> = entries
                    .iter()
                    .map(|&(i, j, value)| Triplet::new(i, j, value / scales[i]))
                    .collect();
                let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
                let assembly_ms = started.elapsed().as_secs_f64() * 1000.;
                let before = Instant::now();
                let symbolic =
                    faer::sparse::linalg::solvers::SymbolicLu::try_new(a.as_ref().symbolic())
                        .unwrap();
                let symbolic_ms = before.elapsed().as_secs_f64() * 1000.;
                let before = Instant::now();
                let lu =
                    faer::sparse::linalg::solvers::Lu::try_new_with_symbolic(symbolic, a.as_ref())
                        .unwrap();
                let numeric_ms = before.elapsed().as_secs_f64() * 1000.;
                let before = Instant::now();
                let mut colnorm = vec![0.; n];
                for &(i, j, value) in &entries {
                    colnorm[j] += (value / scales[i]).abs();
                }
                let condition =
                    colnorm.into_iter().fold(0_f64, f64::max) * inverse_one_norm(&lu, n);
                let condition_ms = before.elapsed().as_secs_f64() * 1000.;
                let rhs = Mat::<f64>::from_fn(n, 1, |i, _| if i == trace { 1. } else { 0. });
                let solution = lu.solve(&rhs);
                let mut output = vec![0.; ext.matrix.len()];
                for (i, &global) in indices.iter().enumerate() {
                    output[global] = solution[(i, 0)];
                }
                let delta = output
                    .iter()
                    .zip(&reference.coefficients)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f64, f64::max);
                let residual = max_abs(&sparse_product(&ext.matrix, &output));
                let accepted =
                    condition.is_finite() && condition < 1e12 && delta < 1e-8 && residual < 1e-5;
                let record = json!({"name":case["name"],"population":population,"state":model.states[population],"assemblyMs":assembly_ms,"symbolicMs":symbolic_ms,"numericMs":numeric_ms,"conditionMs":condition_ms,"totalMs":started.elapsed().as_secs_f64()*1000.,"conditionEstimate":condition,"maxCoefficientDifference":delta,"residual":residual,"accepted":accepted});
                println!("{record}");
                records.push(record);
            }
        }
        std::fs::write(
            folder.join("trace-row-profile.json"),
            serde_json::to_string_pretty(&records).unwrap(),
        )
        .unwrap();
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
    indices:Vec<usize>,
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
struct Trajectory {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    basis_order: usize,
    convergence_error: f64,
}

fn spectral_action(
    lap: &Lapack,
    a: &[f64],
    initial: &[f64],
    times: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let r = initial.len();
    let spec = match spectral(lap, a, r) {
        Ok(spec) => spec,
        Err(_) => return crate::dense_expm::action(lap, a, initial, times),
    };
    let coefficients = lap.solve_factored(&spec.lu, &spec.pivots, initial, r)?;
    let mut modes = vec![0.; r * times.len()];
    for (ti, &t) in times.iter().enumerate() {
        let mut j = 0;
        while j < r {
            if spec.wi[j] > 0. {
                let (s, c) = (spec.wi[j] * t).sin_cos();
                let decay = (spec.wr[j] * t).exp();
                modes[j + ti * r] = decay * (c * coefficients[j] + s * coefficients[j + 1]);
                modes[j + 1 + ti * r] = decay * (-s * coefficients[j] + c * coefficients[j + 1]);
                j += 2;
            } else {
                modes[j + ti * r] = coefficients[j] * (spec.wr[j] * t).exp();
                j += 1;
            }
        }
    }
    let result = lap.multiply(&spec.vectors, &modes, r, times.len(), r);
    if result.iter().any(|x| !x.is_finite()) {
        return Err(
            "Matrix exponential produced a non-finite density; no result is returned".into(),
        );
    }
    Ok((0..times.len())
        .map(|i| {
            if times[i] == 0. {
                initial.to_vec()
            } else {
                result[i * r..(i + 1) * r].to_vec()
            }
        })
        .collect())
}
/// Exact constant-forcing lift: [v;1]'=[[A,b],[0,0]][v;1]. This does not
/// invert A or discard a zero mode, and remains valid for a nonstationary
/// reference and for v(0)=0. The usual eigenbasis/Pade fallback is retained.
fn affine_spectral_action(lap:&Lapack,a:&[f64],initial:&[f64],forcing:&[f64],times:&[f64])->Result<Vec<Vec<f64>>,String>{
    let n=initial.len();
    if a.len()!=n*n||forcing.len()!=n{return Err("Affine exponential dimensions differ".into());}
    let size=n+1;let mut augmented=vec![0.;size*size];
    for j in 0..n{augmented[j*size..j*size+n].copy_from_slice(&a[j*n..(j+1)*n]);}
    augmented[n*size..n*size+n].copy_from_slice(forcing);
    let mut start=initial.to_vec();start.push(1.);
    let result=spectral_action(lap,&augmented,&start,times)?;
    result.into_iter().map(|mut x|{
        let constant=x.pop().unwrap();
        if !constant.is_finite()||(constant-1.).abs()>1e-9{return Err("Affine exponential failed to preserve the constant coordinate".into());}
        Ok(x)
    }).collect()
}
fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn normv(a: &[f64]) -> f64 {
    dotv(a, a).sqrt()
}
fn inverse_one_norm(lu: &faer::sparse::linalg::solvers::Lu<usize, f64>, n: usize) -> f64 {
    inverse_one_norm_with(n, |rhs, transpose| {
        if transpose {
            lu.solve_transpose(rhs)
        } else {
            lu.solve(rhs)
        }
    })
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
type StationaryFactory<'a>=Box<dyn FnOnce(&Lapack,&Sparse)->Result<crate::block_solve::Prepared,String>+'a>;
fn block_stationary_solve(
    lap: &Lapack,
    entries: &[(usize, usize, f64)],
    n: usize,
    trace: usize,
    central: &[usize],
    omega: f64,
    require_condition: bool,
    products:Option<(crate::block_solve::RawProduct,crate::block_solve::RawProduct)>,
    reuse_layout:Option<RefinementLayout>,
    custom_factory:Option<StationaryFactory<'_>>,
) -> Result<StationaryLinearSolution, String> {
    let started = Instant::now();
    let mut a = vec![Vec::new(); n];
    for &(i, j, v) in entries {
        a[j].push((i, v));
    }
    compress(&mut a);
    let schur=SHARED_SCHUR.with(|s|s.borrow().clone()).filter(|s|central.len()==s.d&&n%s.d==0);
    let custom=custom_factory.and_then(|factory|match factory(lap,&a){
        Ok(prepared)=>Some(prepared),
        Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Stationary Hilbert preconditioner unavailable; original factor path retained: {error}");}None},
    });
    let extended=if custom.is_some(){None}else{reuse_layout.as_ref().and_then(|layout|MEAN_REFINEMENT_PRECONDITIONER.with(|cache|{
        let entry=cache.borrow();let previous=entry.as_ref()?;
        let attempt=(||{
            if layout.coordinates.get(trace)!=Some(&previous.trace){return Err("Stationary normalization row changed".into());}
            let embedding=refinement_embedding(&previous.layout,layout,0.,0.,0.,0.)?;
            let factor=previous.prepared.raw_preconditioner().ok_or("Previous stationary factor is unavailable or retired")?;
            crate::block_solve::Prepared::extend_raw(lap,&a,&previous.matrix,factor,&embedding,0.02*omega)
        })();
        match attempt{
            Ok(prepared)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Stationary ILUTP extension {} -> {n} coordinates (same normalization row)",previous.matrix.len());}Some(prepared)},
            Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Stationary ILUTP extension unavailable: {error}");}None},
        }
    }))};
    let prepared = if let Some(prepared)=custom{prepared}else if let Some(prepared)=extended{prepared}else if let Some(s)=schur {crate::block_solve::Prepared::new_schur(lap,&a,s,omega,0.,false)?}else{crate::block_solve::Prepared::new(lap, &a, central, 0.02 * omega)?};
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
    if let Some(layout)=reuse_layout{if prepared.raw_preconditioner().is_some(){
        let trace=layout.coordinates.get(trace).ok_or("Stationary normalization coordinate is missing")?.clone();
        MEAN_REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=Some(ReusableStationary{
            layout,trace,matrix:a,prepared:std::rc::Rc::clone(&prepared),
        }));
    }}
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
fn numerical_poles(model: &Model, ext: &Extended, shift: f64) -> Vec<f64> {
    let n = model.dimension;
    let mut candidates = vec![0.];
    let mut coupling = shift;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                coupling = coupling.max(model.h_real[i * n + j].hypot(model.h_imag[i * n + j]));
            }
        }
    }
    for drive in &model.drives {
        for value in drive
            .cos
            .real
            .iter()
            .chain(&drive.cos.imag)
            .chain(&drive.sin.real)
            .chain(&drive.sin.imag)
        {
            coupling = coupling.max(value.abs());
        }
    }
    for h in &ext.harmonics {
        let offset: f64 = h
            .index
            .iter()
            .zip(&ext.frequency_basis.frequencies)
            .map(|(&i, &frequency)| i as f64 * frequency)
            .sum();
        for i in 0..n {
            for j in 0..n {
                let gap = model.h_real[i * n + i] - model.h_real[j * n + j];
                candidates.push((gap + offset).abs());
                candidates.push((gap - offset).abs());
            }
        }
    }
    candidates.sort_by(f64::total_cmp);
    let mut clusters: Vec<(f64, f64)> = vec![];
    for value in candidates {
        if let Some(last) = clusters.last_mut() {
            if value - last.1 < 8. * coupling {
                last.1 = value;
                continue;
            }
        }
        clusters.push((value, value));
    }
    // These are numerical resolvent shifts, not physical frequency rounding or
    // a discarded-mode threshold. All matrix entries remain in the operator.
    clusters
        .into_iter()
        .map(|(lo, hi)| if lo == 0. { 0. } else { 0.5 * (lo + hi) })
        .collect()
}
fn absolute_numerical_poles(value:&str)->Result<Vec<f64>,String>{
    let values=value.split(',').map(|part|part.trim().parse::<f64>()
        .map_err(|_|"Invalid ATOM_KRYLOV_POLES_RAD numerical pole".to_string()))
        .collect::<Result<Vec<_>,_>>()?;
    if values.is_empty()||values.iter().any(|v|!v.is_finite()){
        return Err("ATOM_KRYLOV_POLES_RAD requires finite angular frequencies in rad/us".into());
    }
    // Signed frequencies are useful for algebra checks; for a real matrix,
    // one sign already contributes BOTH real/imaginary conjugate directions.
    Ok(values)
}
fn fourier_weights(ext:&Extended,n:usize)->Vec<f64>{
    let mut weights=vec![1.;ext.matrix.len()];
    for h in &ext.harmonics{for part in 0..if h.imaginary{2}else{1}{for i in 0..ext.physical_dimension{
        weights[h.offset+part*ext.physical_dimension+i]=(if i<n{1_f64}else{2.}*if h.imaginary{2.}else{1.}).sqrt();
    }}}weights
}
fn normalized_stationary_operator(ext:&Extended,indices:&[usize],n:usize)->Result<(Sparse,usize),String>{
    let mut positions=vec![usize::MAX;ext.matrix.len()];
    for(i,&global)in indices.iter().enumerate(){if global>=positions.len()||positions[global]!=usize::MAX{return Err("Invalid zero-pole active indices".into());}positions[global]=i;}
    let trace=indices.iter().position(|&global|global<n).ok_or("Zero-pole gauge has no central trace coordinate")?;
    let mut a=vec![vec![];indices.len()];
    for(j,&global)in indices.iter().enumerate(){for &(row,value)in &ext.matrix[global]{
        let local=positions[row];if local==usize::MAX&&value!=0.{return Err("Zero-pole active space is not invariant under the complete generator".into());}
        if local!=usize::MAX&&local!=trace{a[j].push((local,value));}
    }if global<n{a[j].push((trace,1.));}}
    compress(&mut a);Ok((a,trace))
}
struct ZeroGaugePrepared{
    layout:RefinementLayout,normalized:Sparse,trace:usize,
    prepared:std::rc::Rc<crate::block_solve::Prepared>,condition:f64,
}
struct GaugeResolvent{
    operator:std::rc::Rc<ZeroGaugePrepared>,weights:Vec<f64>,trace_groups:Vec<Vec<usize>>,
    residual_scales:Vec<f64>,operator_norm:f64,setup_ms:f64,
}
impl GaugeResolvent{
    fn new(lap:&Lapack,model:&Model,_g:&Generator,ext:&Extended,indices:&[usize],weights:&[f64],a:&Sparse,mean:&Stationary)->Result<Self,String>{
        let started=Instant::now();
        if !mean.valid||!mean.condition.is_finite()||mean.condition>1e12||!stationary_numerics_pass(mean){
            return Err("Zero pole requires a validated unique normalized mean on the active operator graph".into());
        }
        let seed_solver=mean.prepared.as_ref().ok_or("The certified mean has no reusable Prepared handle")?;
        if weights!=fourier_weights(ext,model.dimension).as_slice()||a.len()!=indices.len(){return Err("Zero-pole weights or active dimension differ".into());}
        let (normalized,trace)=normalized_stationary_operator(ext,indices,model.dimension)?;
        let layout=refinement_layout(ext,indices,&vec![1.;ext.matrix.len()])?;
        let seed_layout=refinement_layout(&mean.extended,&mean.indices,&vec![1.;mean.extended.matrix.len()])?;
        let(seed_matrix,seed_trace)=normalized_stationary_operator(&mean.extended,&mean.indices,model.dimension)?;
        if !seed_solver.matches_original(&seed_matrix){return Err("The certified mean handle does not match its complete normalized operator".into());}
        let seed=std::rc::Rc::new(ZeroGaugePrepared{layout:seed_layout,normalized:seed_matrix,trace:seed_trace,
            prepared:std::rc::Rc::clone(seed_solver),condition:mean.condition});
        let operator=if layout==seed.layout{
            if normalized!=seed.normalized||trace!=seed.trace{return Err("Same-order zero-pole operator differs from the certified mean".into());}
            ZERO_POLE_REPORT.with(|report|report.borrow_mut().reused_orders+=1);seed
        }else{
            let cached=ZERO_POLE_PREPARED.with(|cache|cache.borrow().clone());
            let parent=cached.filter(|entry|entry.layout.coordinates.len()<layout.coordinates.len()
                &&entry.layout.coordinates.len()>=seed.layout.coordinates.len()).unwrap_or(seed);
            let embedding=refinement_embedding(&parent.layout,&layout,0.,0.,0.,0.)?;
            if embedding[parent.trace]!=trace{return Err("Zero-pole normalization pivot changed across harmonic refinement".into());}
            let raw=parent.prepared.raw_preconditioner().unwrap_or_else(||crate::block_solve::RawPreconditioner::from_shared(std::rc::Rc::clone(&parent.prepared)));
            let omega=ext.frequency_basis.frequencies.iter().copied().fold(f64::INFINITY,f64::min);
            let prepared=crate::block_solve::Prepared::extend_raw(lap,&normalized,&parent.normalized,raw,&embedding,0.02*omega)?;
            let(forward,transpose)=structured_stationary_products(model,ext,indices,trace)?;
            let prepared=std::rc::Rc::new(prepared.with_raw_product(forward).with_raw_transpose(transpose));
            // This is a NEW operator: independently verify its inverse/condition.
            // Do not inherit a lower-K certificate or solve/publish a new mean.
            let condition=block_condition_estimate(lap,&prepared)?;
            ZERO_POLE_REPORT.with(|report|report.borrow_mut().extended_orders+=1);
            std::rc::Rc::new(ZeroGaugePrepared{layout,normalized,trace,prepared,condition})
        };
        if !operator.condition.is_finite()||operator.condition>1e12{return Err("Zero-pole normalized operator is singular or unresolved".into());}
        let mut positions=vec![usize::MAX;ext.matrix.len()];for(i,&global)in indices.iter().enumerate(){positions[global]=i;}
        // Confirm that the trajectory operator is exactly the weighted COMPLETE L.
        for(j,&global)in indices.iter().enumerate(){
            let expected:Vec<_>=ext.matrix[global].iter().filter_map(|&(row,value)|(positions[row]!=usize::MAX)
                .then_some((positions[row],value*weights[row]/weights[global]))).collect();
            if a[j]!=expected{return Err("Zero-pole weighted generator/layout mismatch".into());}
        }
        let mut trace_groups=vec![];
        for h in &ext.harmonics{
            if h.imaginary{
                let frequency=h.index.iter().zip(&ext.frequency_basis.frequencies).map(|(&n,&frequency)|n as f64*frequency).sum::<f64>();
                if !frequency.is_finite()||frequency==0.{return Err("Zero pole encountered an additional zero-frequency harmonic trace mode".into());}
            }
            for part in 0..if h.imaginary{2}else{1}{
                let group:Vec<_>=(0..model.dimension).filter_map(|i|{let local=positions[h.offset+part*ext.physical_dimension+i];(local!=usize::MAX).then_some(local)}).collect();
                if !group.is_empty(){trace_groups.push(group);}
            }
        }
        let mut residual_scales=vec![0_f64;indices.len()];for col in a{for &(i,value)in col{residual_scales[i]=residual_scales[i].max(value.abs());}}
        for scale in &mut residual_scales{if *scale==0.{*scale=1.;}}
        let operator_norm=a.iter().map(|col|col.iter().map(|&(i,value)|value.abs()/residual_scales[i]).sum::<f64>()).fold(0.,f64::max);
        let setup_ms=started.elapsed().as_secs_f64()*1000.;ZERO_POLE_REPORT.with(|report|report.borrow_mut().setup_ms+=setup_ms);
        ZERO_POLE_PREPARED.with(|cache|*cache.borrow_mut()=Some(std::rc::Rc::clone(&operator)));
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Constrained zero-pole prepared n={} condition={:e} setup={setup_ms:.3}ms",indices.len(),operator.condition);}
        Ok(Self{operator,weights:indices.iter().map(|&i|weights[i]).collect(),trace_groups,residual_scales,operator_norm,setup_ms})
    }
    fn project_traces(&self,x:&mut[f64]){
        for group in &self.trace_groups{if let Some((&last,others))=group.split_last(){x[last]= -others.iter().map(|&i|x[i]).sum::<f64>();}}
    }
    fn trace_error(&self,x:&[f64])->f64{self.trace_groups.iter().map(|group|group.iter().map(|&i|x[i]/self.weights[i]).sum::<f64>().abs()).fold(0.,f64::max)}
    fn residual_passes(&self,residual:&[f64],rhs:&[f64],x:&[f64])->bool{
        if residual.iter().chain(x).any(|value|!value.is_finite()){return false;}
        let b:Vec<_>=rhs.iter().zip(&self.residual_scales).map(|(value,scale)|value/scale).collect();
        let r:Vec<_>=residual.iter().zip(&self.residual_scales).map(|(value,scale)|value/scale).collect();
        let tolerance=(1e-11*normv(&b).max(1e-30)).max(32.*f64::EPSILON*(self.operator_norm*x.iter().map(|v|v.abs()).sum::<f64>()+b.iter().map(|v|v.abs()).sum::<f64>()));
        tolerance.is_finite()&&normv(&r)<tolerance
    }
    fn apply(&self,lap:&Lapack,a:&Sparse,v:&[f64])->Result<Vec<f64>,String>{
        if v.len()!=self.weights.len()||a.len()!=v.len()||v.iter().any(|value|!value.is_finite()){
            return Err("Invalid constrained zero-pole RHS".into());
        }
        let input_scale=v.iter().map(|v|v.abs()).sum::<f64>().max(1e-30);
        if self.trace_error(v)>1e-10*input_scale{return Err("Zero-pole RHS is outside the complete traceless dynamic subspace".into());}
        let mut residual=v.to_vec();let mut result=vec![0.;v.len()];
        for _ in 0..2{
            let mut rhs:Vec<_>=residual.iter().zip(&self.weights).map(|(value,weight)|-value/weight).collect();
            rhs[self.operator.trace]=0.;
            let correction=self.operator.prepared.solve(lap,&rhs)?;
            for((value,delta),weight)in result.iter_mut().zip(correction).zip(&self.weights){*value+=delta*weight;}
            // Exact invariant-coordinate lifting, followed by an independent
            // COMPLETE −L residual check INCLUDING the replaced trace row.
            self.project_traces(&mut result);
            let product=sparse_product(a,&result);residual=v.iter().zip(product).map(|(b,value)|b+value).collect();
            if self.residual_passes(&residual,v,&result)&&self.trace_error(&result)<=1e-10*result.iter().map(|v|v.abs()).sum::<f64>().max(1e-30){return Ok(result);}
        }
        Err("Constrained zero-pole inverse failed the complete −L residual or harmonic trace check".into())
    }
    fn closed_space_is_invariant(&self,basis:&[Vec<f64>],products:&[f64],projected:&[f64])->bool{
        let m=basis.len();let n=self.weights.len();
        for j in 0..m{
            let rhs=&products[j*n..(j+1)*n];let mut defect=rhs.to_vec();
            for i in 0..m{for(k,value)in defect.iter_mut().enumerate(){*value-=basis[i][k]*projected[i+j*m];}}
            if !self.residual_passes(&defect,rhs,&basis[j]){return false;}
        }true
    }
}

struct Resolvent {
    lu: Option<faer::sparse::linalg::solvers::Lu<usize, faer::c64>>,
    real_lu: RefCell<Option<faer::sparse::linalg::solvers::Lu<usize, f64>>>,
    real_matrix: Option<SparseColMat<usize, f64>>,
    block: Option<crate::block_solve::Prepared>,
    scales: Vec<f64>,
    augmented:bool,
    reuse_identity:Option<std::rc::Rc<()>>,
    refreshed_reuse:Cell<bool>,
}
impl Resolvent {
    fn new(
        lap: &Lapack,
        a: &Sparse,
        shift: f64,
        frequency: f64,
        central: &[usize],
        omega: f64,
        custom: Option<crate::block_solve::CustomApply>,
        coarse_seeds:&[usize],
        reuse_layout:Option<RefinementLayout>,
    ) -> Result<Self, String> {
        Self::new_with_real_augmented(lap,a,shift,frequency,central,omega,custom,coarse_seeds,reuse_layout,
            std::env::var("ATOM_COMPLEX_REAL_ILUTP").is_ok_and(|value|value=="1"))
    }
    // Explicit policy argument keeps focused tests independent of process
    // environment. The augmented system is exactly the same complex pole.
    fn new_with_real_augmented(
        lap:&Lapack,a:&Sparse,shift:f64,frequency:f64,central:&[usize],omega:f64,
        custom:Option<crate::block_solve::CustomApply>,coarse_seeds:&[usize],
        reuse_layout:Option<RefinementLayout>,real_augmented:bool,
    )->Result<Self,String>{
        let augmented=frequency!=0.&&(custom.is_some()||real_augmented);let original_r=a.len();let r=original_r*if augmented{2}else{1};
        let mut scales = vec![shift.hypot(frequency); r];
        for col in a {
            for &(row, v) in col {
                scales[row] = scales[row].max(v.abs());
            }
        }
        if augmented{for i in 0..original_r{scales[i+original_r]=scales[i];}}
        if frequency == 0. || augmented {
            let mut original: Sparse = a
                .iter()
                .map(|col| col.iter().map(|&(i, v)| (i, -v)).collect())
                .collect();
            for (i, col) in original.iter_mut().enumerate() {
                col.push((i, shift));
            }
            if augmented{let imaginary:Sparse=original.iter().enumerate().map(|(j,col)|{let mut entries:Vec<_>=col.iter().map(|&(i,v)|(i+original_r,v)).collect();entries.push((j,-frequency));entries}).collect();for(j,col)in original.iter_mut().enumerate(){col.push((j+original_r,frequency));}original.extend(imaginary);}
            compress(&mut original);
            let custom=custom.map(|base|{
                if std::env::var_os("ATOM_COARSE_PRECONDITIONER").is_none(){return base;}
                let shared:std::rc::Rc<dyn Fn(&Lapack,&[f64])->Result<Vec<f64>,String>>=std::rc::Rc::from(base);
                let cloned=shared.clone();let mut seeds=coarse_seeds.to_vec();if augmented{seeds.extend(coarse_seeds.iter().map(|&i|i+original_r));}
                match crate::coarse_preconditioner::Prepared::new(lap,&original,Box::new(move|lap,b|cloned(lap,b)),&seeds){
                    Ok(prepared)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Coarse preconditioner rank={} condition={:e}",prepared.rank(),prepared.condition());}Box::new(move|lap:&Lapack,b:&[f64]|prepared.apply(lap,b))as crate::block_solve::CustomApply},
                    Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Coarse preconditioner fallback: {error}");}Box::new(move|lap:&Lapack,b:&[f64]|shared(lap,b))as crate::block_solve::CustomApply}
                }
            });
            let schur=SHARED_SCHUR.with(|s|s.borrow().clone()).filter(|s|!augmented&&central.len()==s.d&&r%s.d==0);
            let selected_central = crate::block_solve::slow_central(&original, central, omega);
            let central = if std::env::var_os("ATOM_DENSE_CENTRAL").is_some(){central}else{&selected_central};
            let extended=if frequency==0.&&custom.is_none(){reuse_layout.as_ref().and_then(|layout|{
                REFINEMENT_PRECONDITIONER.with(|cache|{
                    let cached=cache.borrow();let old=cached.as_ref()?;
                    let attempt=(||{
                        let embedding=refinement_embedding(&old.layout,layout,old.shift,shift,old.frequency,frequency)?;
                        crate::block_solve::Prepared::extend_raw(lap,&original,&old.matrix,std::rc::Rc::clone(&old.factor),&embedding,0.02*omega)
                    })();
                    match attempt{
                        Ok(prepared)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic ILUTP extension {} -> {r} coordinates",old.matrix.len());}Some(prepared)},
                        Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic ILUTP extension unavailable: {error}");}None},
                    }
                })
            })}else{None};
            let block =
                if let Some(custom)=custom {Some(crate::block_solve::Prepared::with_custom(&original,custom)?)}
                else if extended.is_some(){extended}
                else if (full_drive_mode()||real_augmented) && std::env::var_os("ATOM_NO_BLOCK_RESOLVENT").is_none() {
                    if let Some(s)=schur{Some(crate::block_solve::Prepared::new_schur(lap,&original,s,omega,shift,true)?)}else{crate::block_solve::Prepared::new(lap, &original, central, 0.02 * omega).ok()}
                } else {
                    None
                };
            let reuse_identity=if frequency==0.{reuse_layout.and_then(|layout|block.as_ref().and_then(|prepared|prepared.raw_preconditioner()).map(|factor|{
                let identity=std::rc::Rc::new(());REFINEMENT_PRECONDITIONER.with(|cache|*cache.borrow_mut()=Some(ReusableResolvent{
                    identity:std::rc::Rc::clone(&identity),layout,shift,frequency,matrix:original.clone(),factor,
                }));identity
            }))}else{None};
            // Do not assemble/factor the fallback when the iterative path works.
            let matrix = if block.is_none(){
                let mut entries=Vec::new();for(j,col)in original.iter().enumerate(){for &(i,v)in col{entries.push(Triplet::new(i,j,v/scales[i]));}}
                Some(SparseColMat::<usize,f64>::try_new_from_triplets(r,r,&entries).map_err(|e|format!("Real shifted matrix: {e:?}"))?)
            }else{None};
            let lu=matrix.as_ref().map(|m|m.sp_lu().map_err(|e|format!("Real shifted factorization: {e:?}"))).transpose()?;
            return Ok(Self {
                lu: None,
                real_lu: RefCell::new(lu),
                real_matrix: matrix,
                block,
                scales,
                augmented,
                reuse_identity,refreshed_reuse:Cell::new(false),
            });
        }
        let mut entries = vec![];
        for (j, col) in a.iter().enumerate() {
            for &(i, v) in col {
                entries.push(Triplet::new(i, j, faer::c64::new(-v / scales[i], 0.)));
            }
            entries.push(Triplet::new(
                j,
                j,
                faer::c64::new(shift / scales[j], frequency / scales[j]),
            ));
        }
        let matrix = SparseColMat::<usize, faer::c64>::try_new_from_triplets(r, r, &entries)
            .map_err(|e| format!("Complex shifted matrix: {e:?}"))?;
        let lu = matrix
            .sp_lu()
            .map_err(|e| format!("Complex shifted factorization: {e:?}"))?;
        Ok(Self {
            lu: Some(lu),
            real_lu: RefCell::new(None),
            real_matrix: None,
            block: None,
            scales,
            augmented:false,
            reuse_identity:None,refreshed_reuse:Cell::new(false),
        })
    }
    fn apply(&self, lap: &Lapack, v: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        if self.augmented{let n=v.len();let mut rhs=v.to_vec();rhs.resize(2*n,0.);let(real,_)=self.apply_core(lap,&rhs)?;return Ok((real[..n].to_vec(),real[n..].to_vec()));}self.apply_core(lap,v)
    }
    fn apply_core(&self, lap: &Lapack, v: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let _probe_role=crate::matrix_probe::enter_role("trajectory-resolvent");
        let r = v.len();
        if let Some(block) = &self.block {
            if self.real_lu.borrow().is_none(){
            match block.solve(lap, v) {
                Ok(x) => {
                    if block.has_fresh_native()&&!self.refreshed_reuse.replace(true){
                        if let(Some(identity),Some(factor))=(&self.reuse_identity,block.raw_preconditioner()){
                            REFINEMENT_PRECONDITIONER.with(|cache|{if let Some(entry)=cache.borrow_mut().as_mut(){
                                if std::rc::Rc::ptr_eq(identity,&entry.identity){entry.factor=factor;}
                            }});
                        }
                    }
                    return Ok((x, vec![0.; r]));
                },
                Err(e) => {
                    if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                        eprintln!("Block resolvent fallback: {e}");
                    }
                }
            }
            // A failed iterative path is retired for this resolvent. The
            // original scaled equation is factored once, only when needed.
            let entries:Vec<_>=block.matrix.iter().enumerate().flat_map(|(j,col)|col.iter().map(move |&(i,v)|Triplet::new(i,j,v))).collect();
            let matrix=SparseColMat::<usize,f64>::try_new_from_triplets(r,r,&entries).map_err(|e|format!("Real fallback matrix: {e:?}"))?;
            *self.real_lu.borrow_mut()=Some(matrix.sp_lu().map_err(|e|format!("Real fallback LU: {e:?}"))?);
            }
            let rhs=Mat::<f64>::from_fn(r,1,|i,_|v[i]/block.scales[i]);
            let x=self.real_lu.borrow().as_ref().unwrap().solve(&rhs);
            return Ok(((0..r).map(|i|x[(i,0)]).collect(),vec![0.;r]));
        }
        if let Some(matrix) = &self.real_matrix {
            let mut lu = self.real_lu.borrow_mut();
            if lu.is_none() {
                *lu = Some(
                    matrix
                        .sp_lu()
                        .map_err(|e| format!("Real shifted fallback: {e:?}"))?,
                );
            }
            let rhs = Mat::<f64>::from_fn(r, 1, |i, _| v[i] / self.scales[i]);
            let x = lu.as_ref().unwrap().solve(&rhs);
            return Ok(((0..r).map(|i| x[(i, 0)]).collect(), vec![0.; r]));
        }
        let rhs = Mat::<faer::c64>::from_fn(r, 1, |i, _| faer::c64::new(v[i] / self.scales[i], 0.));
        let x = self.lu.as_ref().unwrap().solve(&rhs);
        Ok((
            (0..r).map(|i| x[(i, 0)].re).collect(),
            (0..r).map(|i| x[(i, 0)].im).collect(),
        ))
    }
}

/// Every independent starting vector owns a resolvent chain. Recycled vectors
/// are augmentations, not a pre-existing Arnoldi relation for the new operator.
struct KrylovFrontier {
    jobs: std::collections::VecDeque<(usize, usize)>,
    poles: usize,
    seeds: usize,
    seed_jobs: usize,
    generation_jobs: usize,
    cyclic:bool,
    generation:usize,
    last_index:usize,
    empty_generations:usize,
}
impl KrylovFrontier {
    fn new(seeds: usize, poles: usize) -> Self {
        Self::with_cyclic_poles(seeds,poles,false)
    }
    fn with_cyclic_poles(seeds:usize,poles:usize,cyclic:bool)->Self{
        assert!(seeds>0&&poles>0);
        let mut frontier = Self {
            jobs: std::collections::VecDeque::new(), poles, seeds,
            seed_jobs:seeds*if cyclic{1}else{poles},
            generation_jobs:seeds*if cyclic{1}else{poles},
            cyclic,generation:0,last_index:seeds-1,empty_generations:0,
        };
        if cyclic{frontier.jobs.extend((0..seeds).map(|index|(index,0)));}
        else{for index in 0..seeds { frontier.append(index); }}
        frontier
    }
    fn append(&mut self, index: usize) {
        self.last_index=self.last_index.max(index);
        if self.cyclic{self.jobs.push_back((index,(self.generation+1)%self.poles));}
        else{self.jobs.extend((0..self.poles).map(|pole| (index, pole)));}
    }
    fn next(&mut self) -> Option<(usize, usize)> {
        let job = self.jobs.pop_front()?;
        if job.0 < self.seeds&&(!self.cyclic||self.generation==0) { self.seed_jobs -= 1; }
        self.generation_jobs=self.generation_jobs.saturating_sub(1);
        Some(job)
    }
    fn seeds_expanded(&self) -> bool { self.seed_jobs == 0&&(!self.cyclic||self.generation>=self.poles) }
    fn generation_complete(&self)->bool{self.generation_jobs==0}
    /// Call after appending every child of the final job in this generation.
    fn begin_next_generation(&mut self){
        debug_assert!(self.generation_complete());
        if self.cyclic{
            self.generation+=1;
            if self.jobs.is_empty(){
                self.empty_generations+=1;
                // Near-closure at one pole is not enough to retire the
                // block. Probe every existing direction at the next pole;
                // only an entire empty cycle closes this numerical space.
                if self.empty_generations<self.poles{
                    self.jobs.extend((0..=self.last_index).map(|index|(index,self.generation%self.poles)));
                }
            }else{self.empty_generations=0;}
        }
        self.generation_jobs=self.jobs.len();
    }
}
fn frontier_projection_due(size:usize,checked:usize,interval:usize,completed:bool,closed:bool,limit:usize)->bool{
    (completed&&(size>=checked.saturating_add(interval)||size%interval==0||closed))||size==limit
}

#[cfg(test)]
mod cyclic_pole_tests{
    use super::*;
    fn fixture()->(Sparse,Vec<f64>){
        let n=6;let mut a=vec![0.;n*n];
        for(i,value)in [-0.3,-0.5,-0.8,-0.4,-0.7,-1.1].into_iter().enumerate(){a[i+i*n]=value;}
        for(i,j,value)in [(1,0,1.7),(0,1,-2.3),(2,1,0.4),(4,3,-3.1),(3,4,2.1),(5,4,0.6)]{a[i+j*n]=value;}
        let sparse=(0..n).map(|j|(0..n).filter_map(|i|(a[i+j*n]!=0.).then_some((i,a[i+j*n]))).collect()).collect();(sparse,a)
    }
    fn resolvent(lap:&Lapack,a:&Sparse,frequency:f64)->Resolvent{
        Resolvent::new_with_real_augmented(lap,a,0.6,frequency,&[],1.,None,&[],None,true).unwrap()
    }
    #[test]
    fn cyclic_pole_real_augmented_matches_independent_complex_oracle_with_signed_frequencies(){
        use crate::lapack::Complex64 as Z;
        let lap=Lapack::load().unwrap();let(a,dense)=fixture();let n=a.len();
        for frequency in [0.,2.4,-2.4,17.]{
            let prepared=resolvent(&lap,&a,frequency);assert_eq!(prepared.augmented,frequency!=0.);
            let matrix:Vec<_>=(0..n*n).map(|k|Z{re:-dense[k]+if k%n==k/n{0.6}else{0.},im:if k%n==k/n{frequency}else{0.}}).collect();
            let(lu,pivots)=lap.complex_factor(&matrix,n).unwrap();
            for column in [0,3,n]{
                let rhs:Vec<_>=(0..n).map(|i|if column<n{if i==column{1.}else{0.}}else{(i as f64+0.4).sin()}).collect();
                let oracle=lap.complex_solve_factored(&lu,&pivots,&rhs.iter().map(|&re|Z{re,im:0.}).collect::<Vec<_>>(),n).unwrap();
                let(real,imag)=prepared.apply(&lap,&rhs).unwrap();
                for i in 0..n{assert!((real[i]-oracle[i].re).abs()<2e-12);assert!((imag[i]-oracle[i].im).abs()<2e-12);}
                let ar=sparse_product(&a,&real);let ai=sparse_product(&a,&imag);
                for i in 0..n{
                    assert!((0.6*real[i]-ar[i]-frequency*imag[i]-rhs[i]).abs()<2e-12);
                    assert!((0.6*imag[i]-ai[i]+frequency*real[i]).abs()<2e-12);
                }
            }
        }
        assert_eq!(absolute_numerical_poles("0, 240, -17").unwrap(),[0.,240.,-17.]);
        assert!(absolute_numerical_poles("0,NaN").is_err());assert!(absolute_numerical_poles("").is_err());
    }
    #[test]
    fn cyclic_pole_generation_expands_all_seeds_and_both_complex_children(){
        let mut work=KrylovFrontier::with_cyclic_poles(2,3,true);
        assert_eq!(work.next(),Some((0,0)));work.append(2);work.append(3);
        assert!(!work.generation_complete());assert!(!work.seeds_expanded());
        assert_eq!(work.next(),Some((1,0)));work.append(4);
        assert!(work.generation_complete());work.begin_next_generation();assert!(!work.seeds_expanded());
        for(source,child)in [(2,5),(3,6),(4,7)]{
            assert_eq!(work.next(),Some((source,1)));work.append(child);
            if source<4{assert!(!work.generation_complete());}
        }
        assert!(work.generation_complete());work.begin_next_generation();assert!(!work.seeds_expanded());
        for(source,child)in [(5,8),(6,9),(7,10)]{assert_eq!(work.next(),Some((source,2)));work.append(child);}
        assert!(work.generation_complete());work.begin_next_generation();assert!(work.seeds_expanded());
        assert_eq!(work.next(),Some((8,0)));assert!(!work.generation_complete());
        assert!(!frontier_projection_due(32,16,16,work.generation_complete(),false,768));
    }
    #[test]
    fn cyclic_pole_near_closed_space_checks_every_pole_before_retiring_any_seed(){
        let mut work=KrylovFrontier::with_cyclic_poles(2,3,true);
        // Even if no new direction is found at the first pole, both seeds
        // must be revisited under the remaining poles before closure.
        for pole in 0..3{
            assert_eq!(work.next(),Some((0,pole)));assert_eq!(work.next(),Some((1,pole)));
            assert!(work.generation_complete());work.begin_next_generation();
            if pole<2{assert!(!work.jobs.is_empty());assert!(!work.seeds_expanded());}
        }
        assert!(work.jobs.is_empty());assert!(work.seeds_expanded());assert_eq!(work.next(),None);
    }
    #[test]
    fn cyclic_pole_two_seed_affine_projection_matches_complete_dense_evolution(){
        let lap=Lapack::load().unwrap();let(a,dense)=fixture();let n=a.len();
        let unit=|i:usize|{let mut x=vec![0.;n];x[i]=1.;x};let initial=unit(0);let forcing=unit(3);
        let poles=[resolvent(&lap,&a,2.4),resolvent(&lap,&a,0.)];
        let mut basis=vec![initial.clone(),forcing.clone()];let mut work=KrylovFrontier::with_cyclic_poles(2,poles.len(),true);
        let mut complex_children=0;
        for _ in 0..64{
            let Some((source,pole))=work.next()else{break;};
            let(real,imag)=poles[pole].apply(&lap,&basis[source]).unwrap();
            for(mut value,is_imag)in [(real,false),(imag,true)]{
                for _ in 0..2{for q in &basis{let dot=dotv(q,&value);for(x,&q)in value.iter_mut().zip(q){*x-=dot*q;}}}
                let length=normv(&value);if length>1e-12{
                    if is_imag{complex_children+=1;}
                    let index=basis.len();basis.push(value.into_iter().map(|v|v/length).collect());work.append(index);
                }
            }
            if work.generation_complete(){work.begin_next_generation();if basis.len()==n&&work.seeds_expanded(){break;}}
        }
        assert_eq!(basis.len(),n,"both initial and forcing chains must span their independent blocks");
        assert!(complex_children>0,"the complex solve's imaginary child must participate");
        let mut projected=vec![0.;n*n];for j in 0..n{let aq=sparse_product(&a,&basis[j]);for i in 0..n{projected[i+j*n]=dotv(&basis[i],&aq);}}
        let start:Vec<_>=basis.iter().map(|q|dotv(q,&initial)).collect();let force:Vec<_>=basis.iter().map(|q|dotv(q,&forcing)).collect();
        let times=[0.,0.11,0.7,3.2];let small=affine_spectral_action(&lap,&projected,&start,&force,&times).unwrap();
        let mut augmented=vec![0.;(n+1)*(n+1)];for j in 0..n{for i in 0..n{augmented[i+j*(n+1)]=dense[i+j*n];}}
        for i in 0..n{augmented[i+n*(n+1)]=forcing[i];}let mut augmented_initial=initial;augmented_initial.push(1.);
        let oracle=crate::dense_expm::action(&lap,&augmented,&augmented_initial,&times).unwrap();
        for(values,expected)in small.iter().zip(oracle){let mut actual=vec![0.;n];for(j,&c)in values.iter().enumerate(){for(i,x)in actual.iter_mut().enumerate(){*x+=basis[j][i]*c;}}
            assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-11));
        }
    }
}

#[cfg(test)]
mod krylov_frontier_tests {
    use super::*;
    #[test]
    fn complete_generation_spacing_does_not_skip_the_minimum_basis_boundary(){
        let mut checked=0;let mut checks=vec![];
        for size in (2..=64).step_by(2){let interval=if size<32{8}else{16};
            if frontier_projection_due(size,checked,interval,true,false,768){checks.push(size);checked=size;}
        }
        assert_eq!(checks,[8,16,24,32,48,64]);
        assert!(!frontier_projection_due(32,24,16,false,false,768));
    }

    #[test]
    fn all_starting_chains_are_expanded_instead_of_only_the_last_recycled_vector() {
        // R=(shift I-A)^-1 has two disjoint chains. The observable on e1 is
        // driven by the actual initial e0, while e2 is an unrelated recycle.
        // Extending only e2's chain keeps every successive projected answer
        // unchanged while completely missing the e0 -> e1 response.
        fn run(frontier: bool) -> Vec<Vec<f64>> {
            let unit = |i: usize| { let mut v=vec![0.;8]; v[i]=1.; v };
            let mut basis=vec![unit(0),unit(2)];
            let mut work=KrylovFrontier::new(basis.len(),1);
            for _ in 0..16 {
                let source=if frontier { match work.next() {Some((i,_))=>i,None=>break} } else {basis.len()-1};
                let mut value:Vec<_>=basis[source].iter().map(|v|0.5*v).collect();
                value[1]+=0.1*basis[source][0];
                for i in 2..7 {value[i+1]+=0.1*basis[source][i];}
                for _ in 0..2 {for q in &basis {let projection=dotv(q,&value);for (x,&v) in value.iter_mut().zip(q){*x-=projection*v;}}}
                let norm=normv(&value);
                if norm>1e-12 {let index=basis.len();basis.push(value.into_iter().map(|v|v/norm).collect());work.append(index);}
            }
            basis
        }
        let last_only=run(false);let complete=run(true);
        assert!(last_only.iter().all(|q|q[1]==0.));
        assert_eq!(complete.len(),8);
        assert!(complete.iter().any(|q|q[1].abs()==1.));
        // A=I-R^-1 is stable, and its e0 -> e1 response at t=1 is
        // 0.4/e, despite successive last-chain projections agreeing exactly.
        // This tests subspace algebra; it is not a Lindblad physics fixture.
    }

    #[test]
    fn frontier_visits_each_seed_and_each_pole_before_acceptance() {
        let mut frontier=KrylovFrontier::new(2,2);
        assert_eq!(frontier.next(),Some((0,0)));frontier.append(2);
        assert_eq!(frontier.next(),Some((0,1)));assert!(!frontier.seeds_expanded());
        assert_eq!(frontier.next(),Some((1,0)));assert!(!frontier.seeds_expanded());
        assert_eq!(frontier.next(),Some((1,1)));assert!(frontier.seeds_expanded());
        assert_eq!(frontier.next(),Some((2,0)));assert_eq!(frontier.next(),Some((2,1)));
        assert_eq!(frontier.next(),None);
    }

    #[test]
    fn unrelated_recycled_children_cannot_certify_a_still_missing_initial_grandchild(){
        // A stable exact example: R=.5I+.1N, A=I-R^-1. N has the
        // initial chain e0->e1->e2->e3, plus unrelated recycled chains.
        // While e2 is absent, its projected population is identically zero,
        // but the exact t=.5 response is -.02 exp(-.5), over 1%.
        let mut work=KrylovFrontier::new(5,1);
        let seeds=[0usize,4,6,8,10];let mut coordinates=seeds.to_vec();
        let mut plateau_checks=0;
        for seed in 0..5{
            assert_eq!(work.next(),Some((seed,0)));
            coordinates.push(seeds[seed]+1);work.append(coordinates.len()-1);
            if seed>0{plateau_checks+=1;assert!(!coordinates.contains(&2));}
        }
        assert_eq!(plateau_checks,4);assert!(work.seeds_expanded());assert!(work.generation_complete());
        assert!((-0.02_f64*(-0.5_f64).exp()).abs()>0.01);
        work.begin_next_generation();
        // The initial child is expanded first in the next complete sweep.
        assert_eq!(work.next(),Some((5,0)));coordinates.push(2);work.append(coordinates.len()-1);
        assert!(work.seeds_expanded());assert!(!work.generation_complete());
        for child in 6..10{assert_eq!(work.next(),Some((child,0)));}
        assert!(work.generation_complete());assert!(coordinates.contains(&2));
        // A comparison between completed generations now sees the missing
        // response; intermediate unrelated additions cannot produce a gate.
        work.begin_next_generation();assert_eq!(work.next(),Some((10,0)));
    }
}

#[cfg(test)]
mod affine_reference_tests{
    use super::*;
    #[test]
    fn affine_auto_selects_only_long_single_axis_full_manifold_requests(){
        assert!(prefer_affine_center(32,1,100000.,160,false));
        assert!(!prefer_affine_center(32,1,20.,161,false));
        assert!(!prefer_affine_center(32,1,10.,2,false));
        assert!(prefer_affine_center(32,1,10_f64.next_up(),2,false));
        assert!(!prefer_affine_center(32,2,100000.,160,false));
        assert!(!prefer_affine_center(2,1,100000.,160,false));
        assert!(!prefer_affine_center(32,1,100000.,160,true));
        assert!(!prefer_affine_center(32,1,f64::NAN,160,false));
        assert!(!prefer_affine_center(32,1,100000.,1,false));
    }
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
    fn affine_nonstationary_reference_and_zero_transient_match_direct_full_density(){
        let lap=Lapack::load().unwrap();let model=driven_two_level();let g=build_generator(&model).unwrap();let ext=extend(&model,&g,2).unwrap();
        let options=Options{duration_us:1.7,samples:9,include_asymptotic:false,..Options::default()};
        let times:Vec<_>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
        let n=ext.matrix.len();let mut dense=vec![0.;n*n];
        for(j,col)in ext.matrix.iter().enumerate(){for &(i,value)in col{dense[i+j*n]=value;}}
        // Independent direct full-generator Pade evolution, with no center.
        let oracle=crate::dense_expm::action(&lap,&dense,&ext.initial,&times).unwrap();
        let arbitrary:Vec<_>=(0..n).map(|j|0.13*(0.7+j as f64).sin()).collect();
        for reference in [&arbitrary,&ext.initial]{
            assert!(max_abs(&sparse_product(&ext.matrix,reference))>0.01);
            let actual=trajectory_from_reference(&lap,&model,&g,&ext,reference,false,true,&options).unwrap();
            for((state,expected),&time)in actual.states.iter().zip(&oracle).zip(&times){
                let expected=reconstruct(&ext,expected,&[model.drives[0].omega*time]);
                assert!(state.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-10));
            }
            assert!(actual.states[0].iter().zip(&g.initial).all(|(a,b)|(a-b).abs()<2e-14));
        }
    }
    #[test]
    fn affine_exponential_retains_forcing_in_singular_zero_modes(){
        let lap=Lapack::load().unwrap();let a=vec![0.,0.,0.,-2.];let initial=vec![0.,0.];let forcing=vec![0.3,0.8];
        let times=[0.,0.2,3.];let result=affine_spectral_action(&lap,&a,&initial,&forcing,&times).unwrap();
        for(x,&t)in result.iter().zip(&times){assert!((x[0]-0.3*t).abs()<1e-12);assert!((x[1]-0.4*(1.-(-2.*t).exp())).abs()<1e-12);}
    }
    #[test]
    fn reference_lifting_uses_harmonic_indices_when_multifrequency_offsets_move(){
        let mut model=driven_two_level();let mut second=model.drives[0].clone();second.omega=2.3_f64.sqrt();model.drives.push(second);
        let g=build_generator(&model).unwrap();let old=extend(&model,&g,1).unwrap();let new=extend(&model,&g,2).unwrap();
        let coefficients:Vec<_>=(0..old.matrix.len()).map(|j|0.01*(j+1)as f64).collect();
        let lifted=lift_reference(&old,&coefficients,&new).unwrap();
        assert!(old.harmonics.iter().any(|h|new.harmonics.iter().any(|other|h.index==other.index&&h.offset!=other.offset)));
        for phase in [[0.,0.],[0.3,-0.7],[2.1,4.2]]{
            let a=reconstruct(&old,&coefficients,&phase);let b=reconstruct(&new,&lifted,&phase);
            assert!(a.iter().zip(b).all(|(a,b)|(a-b).abs()<1e-14));
        }
    }
    #[test]
    fn refinement_embedding_handles_relocation_new_old_harmonic_coordinates_and_rejects_changed_metadata(){
        let mut model=driven_two_level();let mut second=model.drives[0].clone();second.omega=2.3_f64.sqrt();model.drives.push(second);
        let g=build_generator(&model).unwrap();let old=extend(&model,&g,1).unwrap();let new=extend(&model,&g,2).unwrap();
        let old_indices:Vec<_>=(0..old.matrix.len()).filter(|&i|i!=1).collect();let new_indices:Vec<_>=(0..new.matrix.len()).collect();
        let old_layout=refinement_layout(&old,&old_indices,&vec![1.;old.matrix.len()]).unwrap();
        let new_layout=refinement_layout(&new,&new_indices,&vec![1.;new.matrix.len()]).unwrap();
        let map=refinement_embedding(&old_layout,&new_layout,0.01,0.01,0.,0.).unwrap();
        assert!(map.iter().zip(&old_indices).any(|(new,old)|new!=old));
        assert!(!map.contains(&1)); // new reachable coordinate in the OLD central harmonic
        for(i,&j)in map.iter().enumerate(){assert_eq!(old_layout.coordinates[i],new_layout.coordinates[j]);}
        let mut changed=new_layout.clone();changed.weights[map[0]]=2.;
        assert!(refinement_embedding(&old_layout,&changed,0.01,0.01,0.,0.).is_err());
        assert!(refinement_embedding(&old_layout,&new_layout,0.01,0.02,0.,0.).is_err());
        assert!(refinement_embedding(&old_layout,&new_layout,0.01,0.01,0.,1.).is_err());
        let mut changed=new_layout;changed.frequencies[0]+=1e-12;
        assert!(refinement_embedding(&old_layout,&changed,0.01,0.01,0.,0.).is_err());
    }
    #[test]
    fn affine_trace_decomposition_changes_only_the_reference_and_retains_its_full_forcing(){
        let model=driven_two_level();let g=build_generator(&model).unwrap();let ext=extend(&model,&g,2).unwrap();
        let reference:Vec<_>=(0..ext.matrix.len()).map(|i|0.11*(i as f64+0.3).cos()).collect();
        let adjusted=trace_matched_reference(&ext,&reference,model.dimension).unwrap();
        let forcing=sparse_product(&ext.matrix,&adjusted);
        for h in &ext.harmonics{for part in 0..if h.imaginary{2}else{1}{let offset=h.offset+part*ext.physical_dimension;
            let target=ext.initial[offset..offset+2].iter().sum::<f64>();
            assert!((adjusted[offset]+adjusted[offset+1]-target).abs()<1e-15);
            assert!((forcing[offset]+forcing[offset+1]).abs()<2e-14);
        }}
        assert!(max_abs(&forcing)>0.01); // nonstationary forcing was recomputed, not removed
        let delta:Vec<_>=ext.initial.iter().zip(&adjusted).map(|(a,b)|a-b).collect();
        let derivative=sparse_product(&ext.matrix,&delta);let exact=sparse_product(&ext.matrix,&ext.initial);
        assert!(derivative.iter().zip(forcing).zip(exact).all(|((a,b),c)|(a+b-c).abs()<2e-14));
    }
}

#[cfg(test)]
mod zero_pole_tests{
    use super::*;
    fn model(rate:f64,periodic:bool)->Model{
        let zero=Operator{real:vec![0.;4],imag:vec![0.;4]};
        Model{dimension:2,states:vec![],h_real:vec![0.,0.3,0.3,1.2],h_imag:vec![0.,0.1,-0.1,0.],
            collapse:vec![Operator{real:vec![0.,rate.sqrt(),0.,0.],imag:vec![0.;4]}],initial_index:0,
            initial_real:None,initial_imag:None,time_dependent:periodic,frequency_coordinates:None,
            drives:if periodic{vec![Drive{omega:2.3,cos:Operator{real:vec![0.,0.01,0.01,0.],imag:vec![0.;4]},sin:zero}]}else{vec![]}}
    }
    fn mean(lap:&Lapack,model:&Model,order:i32)->(Generator,Stationary){
        let g=build_generator(model).unwrap();let mut mean=stationary(lap,model,&g,order).unwrap();
        let(a,_)=normalized_stationary_operator(&mean.extended,&mean.indices,model.dimension).unwrap();
        let central:Vec<_>=mean.indices.iter().enumerate().filter_map(|(i,&global)|(global<mean.extended.physical_dimension).then_some(i)).collect();
        let prepared=std::rc::Rc::new(crate::block_solve::Prepared::new(lap,&a,&central,0.).unwrap());
        mean.condition=block_condition_estimate(lap,&prepared).unwrap();mean.prepared=Some(prepared);
        assert!(stationary_numerics_pass(&mean));(g,mean)
    }
    fn weighted(ext:&Extended,indices:&[usize],weights:&[f64])->Sparse{
        let mut positions=vec![usize::MAX;ext.matrix.len()];for(i,&global)in indices.iter().enumerate(){positions[global]=i;}
        indices.iter().map(|&global|ext.matrix[global].iter().filter_map(|&(row,value)|(positions[row]!=usize::MAX)
            .then_some((positions[row],value*weights[row]/weights[global]))).collect()).collect()
    }
    fn trace_basis(gauge:&GaugeResolvent)->Vec<Vec<f64>>{
        let n=gauge.weights.len();let mut basis:Vec<Vec<f64>>=vec![];
        for i in 0..n{let mut x=vec![0.;n];x[i]=1.;
            for group in &gauge.trace_groups{let value=group.iter().map(|&i|x[i]).sum::<f64>()/group.len()as f64;for &i in group{x[i]-=value;}}
            for _ in 0..2{for q in &basis{let coefficient=dotv(q,&x);for(v,&q)in x.iter_mut().zip(q){*v-=coefficient*q;}}}
            let length=normv(&x);if length>1e-12{basis.push(x.into_iter().map(|v|v/length).collect());}
        }basis
    }
    fn dense_inverse(lap:&Lapack,gauge:&GaugeResolvent,a:&Sparse,v:&[f64])->Vec<f64>{
        let q=trace_basis(gauge);let m=q.len();let mut reduced=vec![0.;m*m];
        for(j,column)in q.iter().enumerate(){let product=sparse_product(a,column);for(i,row)in q.iter().enumerate(){reduced[i+j*m]=-dotv(row,&product);}}
        let rhs:Vec<_>=q.iter().map(|q|dotv(q,v)).collect();let coefficients=lap.solve(&reduced,&rhs,m).unwrap();
        let mut result=vec![0.;v.len()];for(q,&c)in q.iter().zip(&coefficients){for(x,&q)in result.iter_mut().zip(q){*x+=c*q;}}result
    }
    #[test]
    fn constrained_zero_inverse_matches_independent_dense_traceless_oracle(){
        clear_request_results();let lap=Lapack::load().unwrap();let model=model(0.49,false);let(g,mean)=mean(&lap,&model,1);
        let weights=fourier_weights(&mean.extended,model.dimension);let a=weighted(&mean.extended,&mean.indices,&weights);
        let gauge=GaugeResolvent::new(&lap,&model,&g,&mean.extended,&mean.indices,&weights,&a,&mean).unwrap();
        let v=vec![0.7,-0.7,0.2,-0.3];let expected=dense_inverse(&lap,&gauge,&a,&v);let actual=gauge.apply(&lap,&a,&v).unwrap();
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));assert!(gauge.trace_error(&actual)<1e-13);
        let residual:Vec<_>=sparse_product(&a,&actual).iter().zip(&v).map(|(a,b)|a+b).collect();assert!(max_abs(&residual)<1e-11);
        let q=trace_basis(&gauge);let m=q.len();let projected:Vec<_>=q.iter().flat_map(|column|{let product=sparse_product(&a,column);q.iter().map(move|row|dotv(row,&product))}).collect();
        let reference=vec![0.8,0.2,0.04,-0.03];let forcing=sparse_product(&a,&reference);let initial:Vec<_>=g.initial.iter().enumerate().map(|(i,x)|x*weights[i]-reference[i]).collect();
        let start:Vec<_>=q.iter().map(|q|dotv(q,&initial)).collect();let force:Vec<_>=q.iter().map(|q|dotv(q,&forcing)).collect();
        let times=[0.,0.2,1.7];let reduced=affine_spectral_action(&lap,&projected,&start,&force,&times).unwrap();
        let mut dense=vec![0.;a.len()*a.len()];for(j,col)in a.iter().enumerate(){for &(i,value)in col{dense[i+j*a.len()]=value;}}
        let full_initial:Vec<_>=g.initial.iter().zip(&weights).map(|(x,w)|x*w).collect();let oracle=crate::dense_expm::action(&lap,&dense,&full_initial,&times).unwrap();
        for(row,expected)in reduced.iter().zip(oracle){let mut result=reference.clone();for j in 0..m{for(i,value)in result.iter_mut().enumerate(){*value+=q[j][i]*row[j];}}assert!(result.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));}
    }
    #[test]
    fn zero_pole_higher_harmonic_extension_has_its_own_condition_and_full_residual(){
        clear_request_results();let lap=Lapack::load().unwrap();let model=model(0.49,true);let(g,mean)=mean(&lap,&model,1);
        let weights=fourier_weights(&mean.extended,model.dimension);let a=weighted(&mean.extended,&mean.indices,&weights);
        GaugeResolvent::new(&lap,&model,&g,&mean.extended,&mean.indices,&weights,&a,&mean).unwrap();
        let higher=extend(&model,&g,2).unwrap();let indices=reachable(&higher.matrix,&higher.initial);let weights=fourier_weights(&higher,model.dimension);let a=weighted(&higher,&indices,&weights);
        let gauge=GaugeResolvent::new(&lap,&model,&g,&higher,&indices,&weights,&a,&mean).unwrap();assert!(gauge.operator.condition.is_finite());
        let mut v:Vec<_>=(0..indices.len()).map(|i|(0.23+i as f64).sin()).collect();gauge.project_traces(&mut v);
        let expected=dense_inverse(&lap,&gauge,&a,&v);let actual=gauge.apply(&lap,&a,&v).unwrap();
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-10));assert!(gauge.trace_error(&actual)<1e-13);
        assert_eq!(ZERO_POLE_REPORT.with(|report|report.borrow().extended_orders),1);
        let mut wrong=weights.clone();wrong[model.dimension]*=2.;assert!(GaugeResolvent::new(&lap,&model,&g,&higher,&indices,&wrong,&a,&mean).is_err());
    }
    #[test]
    fn zero_pole_keeps_the_active_graph_and_nonunique_mean_uses_original_fallback(){
        clear_request_results();let lap=Lapack::load().unwrap();let mut active=Model{dimension:3,states:vec![],h_real:vec![0.,0.,0.,0.,0.,0.3,0.,0.3,1.2],h_imag:vec![0.;9],
            collapse:vec![Operator{real:vec![0.,0.,0.,0.,0.,0.7,0.,0.,0.],imag:vec![0.;9]}],initial_index:1,initial_real:None,initial_imag:None,time_dependent:false,drives:vec![],frequency_coordinates:None};
        let(g,mean)=mean(&lap,&active,1);let mut reference=mean.coefficients.clone();reference[1]+=4.*f64::EPSILON;
        let adjusted=trace_matched_reference(&mean.extended,&reference,3).unwrap();assert_eq!(adjusted[0],0.);
        let support:Vec<_>=mean.extended.initial.iter().zip(&adjusted).map(|(&a,&b)|if a!=0.||b!=0.{1.}else{0.}).collect();
        assert_eq!(reachable(&mean.extended.matrix,&support),reachable(&mean.extended.matrix,&mean.extended.initial));
        let weights=fourier_weights(&mean.extended,3);let a=weighted(&mean.extended,&mean.indices,&weights);
        GaugeResolvent::new(&lap,&active,&g,&mean.extended,&mean.indices,&weights,&a,&mean).unwrap();
        active.h_real.fill(0.);active.initial_index=2;active.collapse=vec![Operator{real:vec![0.,0.,0.5,0.,0.,0.,0.,0.,0.],imag:vec![0.;9]},Operator{real:vec![0.,0.,0.,0.,0.,0.8,0.,0.,0.],imag:vec![0.;9]}];
        let g=build_generator(&active).unwrap();let ext=extend(&active,&g,1).unwrap();let unresolved=unresolved_stationary(ext);
        let options=Options{duration_us:2.,samples:3,include_asymptotic:false,..Options::default()};
        let result=trajectory_with_mean_attempt(&lap,&active,&g,&unresolved.extended,&unresolved.coefficients,false,false,&options,&unresolved,true).unwrap();
        let excited=(-0.89_f64*2.).exp();assert!((result.states[2][2]-excited).abs()<1e-12);
        assert!((result.states[2][0]-(1.-excited)*0.25/0.89).abs()<1e-12);
        assert!(ZERO_POLE_DISABLED.with(Cell::get));assert_eq!(ZERO_POLE_REPORT.with(|report|report.borrow().fallbacks),1);
    }
    #[test]
    fn weak_relaxation_zero_inverse_rejects_a_false_closed_subspace(){
        clear_request_results();let lap=Lapack::load().unwrap();let model=model(1e-8,false);let(g,mean)=mean(&lap,&model,1);
        let weights=fourier_weights(&mean.extended,2);let a=weighted(&mean.extended,&mean.indices,&weights);
        let gauge=GaugeResolvent::new(&lap,&model,&g,&mean.extended,&mean.indices,&weights,&a,&mean).unwrap();
        let v=vec![0.7,-0.7,0.2,-0.3];let actual=gauge.apply(&lap,&a,&v).unwrap();let expected=dense_inverse(&lap,&gauge,&a,&v);
        let error=actual.iter().zip(&expected).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);assert!(error<1e-7*max_abs(&expected));
        let q=vec![1./2_f64.sqrt(),-1./2_f64.sqrt(),0.,0.];let product=sparse_product(&a,&q);let projected=vec![dotv(&q,&product)];
        assert!(!gauge.closed_space_is_invariant(&[q],&product,&projected));
        let options=Options{duration_us:100000.,samples:3,include_asymptotic:false,..Options::default()};
        let fallback=trajectory_from_reference(&lap,&model,&g,&mean.extended,&mean.coefficients,true,false,&options).unwrap();
        let mut dense=vec![0.;16];for(j,col)in g.matrix.iter().enumerate(){for &(i,value)in col{dense[i+j*4]=value;}}
        let oracle=crate::dense_expm::action(&lap,&dense,&g.initial,&fallback.times).unwrap();
        assert!(fallback.states.iter().zip(oracle).all(|(row,expected)|row.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-6)));
    }
}

fn trajectory_density_error(lap:&Lapack,g:&Generator,states:&[Vec<f64>])->Result<(f64,f64),String>{
    let mut trace_error=0_f64;let mut positivity_error=0_f64;
    for state in states {
        if state.iter().any(|value|!value.is_finite()){return Ok((f64::INFINITY,f64::INFINITY));}
        trace_error=trace_error.max((state[..g.n].iter().sum::<f64>()-1.).abs());
        positivity_error=positivity_error.max(positivity(lap,&expand_density(&g.coords,state,g.n),g.n)?);
    }
    Ok((trace_error,positivity_error))
}

/// Shift-invert Arnoldi is a numerical matrix-exponential method on the full
/// Fourier generator. Increasing the Krylov space retains more modes; no
/// density coherence is removed from the physical model.
fn trajectory(
    lap: &Lapack,
    model: &Model,
    g: &Generator,
    stationary: &Stationary,
    options: &Options,
) -> Result<Trajectory, String> {
    trajectory_with_mean(lap,model,g,&stationary.extended,&stationary.coefficients,stationary.valid,false,options,stationary)
}
fn trajectory_with_mean(lap:&Lapack,model:&Model,g:&Generator,ext:&Extended,reference:&[f64],traceless_transient:bool,affine:bool,options:&Options,mean:&Stationary)->Result<Trajectory,String>{
    let try_zero=full_drive_mode()&&(traceless_transient||affine)&&!ZERO_POLE_DISABLED.with(Cell::get)
        &&std::env::var("ATOM_ZERO_POLE_MEAN_REUSE").is_ok_and(|value|value=="1");
    trajectory_with_mean_attempt(lap,model,g,ext,reference,traceless_transient,affine,options,mean,try_zero)
}
fn trajectory_with_mean_attempt(lap:&Lapack,model:&Model,g:&Generator,ext:&Extended,reference:&[f64],traceless_transient:bool,affine:bool,options:&Options,mean:&Stationary,try_zero:bool)->Result<Trajectory,String>{
    if try_zero{
        ZERO_POLE_REPORT.with(|report|report.borrow_mut().attempts+=1);
        let attempted=if !mean.valid||!mean.condition.is_finite()||mean.prepared.is_none(){Err("Zero pole has no certified unique mean/factor handle".into())}
            else{trajectory_reference_attempt(lap,model,g,ext,reference,traceless_transient,affine,options,Some(mean))};
        match attempted{
            Ok(result)=>return Ok(result),
            Err(error)=>{
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Zero-pole route rejected; discard its Krylov space and restart the original positive-pole route: {error}");}
                ZERO_POLE_DISABLED.with(|flag|flag.set(true));ZERO_POLE_PREPARED.with(|cache|*cache.borrow_mut()=None);
                RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);
                ZERO_POLE_REPORT.with(|report|{let mut report=report.borrow_mut();report.fallbacks+=1;report.reasons.push(error);});
            }
        }
    }
    trajectory_from_reference(lap,model,g,ext,reference,traceless_transient,affine,options)
}
fn trace_matched_reference(ext:&Extended,reference:&[f64],hilbert_dimension:usize)->Result<Vec<f64>,String>{
    if reference.len()!=ext.matrix.len()||hilbert_dimension==0||hilbert_dimension*hilbert_dimension!=ext.physical_dimension{
        return Err("Affine reference trace coordinates differ".into());
    }
    let mut result=reference.to_vec();
    let support:Vec<_>=ext.initial.iter().zip(reference).map(|(&a,&b)|if a!=0.||b!=0.{1.}else{0.}).collect();
    let mut active=vec![false;ext.matrix.len()];for index in reachable(&ext.matrix,&support){active[index]=true;}
    for h in &ext.harmonics{for part in 0..if h.imaginary{2}else{1}{
        let offset=h.offset+part*ext.physical_dimension;
        let target=ext.initial[offset..offset+hilbert_dimension].iter().sum::<f64>();
        if h.imaginary&&target!=0.{return Err("A nonzero initial harmonic trace requires its explicit rotating invariant".into());}
        let pivot=(offset..offset+hilbert_dimension).find(|&i|active[i]);
        let Some(pivot)=pivot else{
            if target!=result[offset..offset+hilbert_dimension].iter().sum::<f64>(){return Err("Affine trace correction has no active population direction".into());}
            continue;
        };
        let other=(offset..offset+hilbert_dimension).filter(|&i|i!=pivot).map(|i|result[i]).sum::<f64>();
        // c_eff=c+u(Tr rho0-Tr c), with Tr u=1. Recompute the COMPLETE
        // A*c_eff afterward. No evolved density is normalized or repaired.
        // Use an existing active population: a roundoff-size gauge change
        // must not create a new physical graph component.
        result[pivot]=target-other;
    }}
    if result.iter().any(|value|!value.is_finite()){return Err("Non-finite affine reference trace decomposition".into());}
    Ok(result)
}
fn trajectory_from_reference(
    lap:&Lapack,model:&Model,g:&Generator,ext:&Extended,reference:&[f64],
    traceless_transient:bool,affine:bool,options:&Options,
)->Result<Trajectory,String>{
    trajectory_reference_attempt(lap,model,g,ext,reference,traceless_transient,affine,options,None)
}
fn trajectory_reference_attempt(
    lap:&Lapack,model:&Model,g:&Generator,ext:&Extended,reference:&[f64],
    traceless_transient:bool,affine:bool,options:&Options,zero_seed:Option<&Stationary>,
)->Result<Trajectory,String>{
    if reference.len()!=ext.matrix.len(){return Err("Trajectory reference dimensions differ".into());}
    let adjusted_reference=if affine{Some(trace_matched_reference(ext,reference,model.dimension)?)}else{None};
    let reference=adjusted_reference.as_deref().unwrap_or(reference);
    let traceless_transient=traceless_transient||affine;
    let inds = if affine{
        // An arbitrary reference can have support outside the initial orbit.
        // Its transient cancels that support; retain the complete union so
        // neither half of the affine identity is silently removed.
        let support:Vec<_>=ext.initial.iter().zip(reference).map(|(&a,&b)|if a!=0.||b!=0.{1.}else{0.}).collect();
        reachable(&ext.matrix,&support)
    }else{reachable(&ext.matrix, &ext.initial)};
    let r = inds.len();
    let mut positions = vec![usize::MAX; ext.matrix.len()];
    for (i, &global) in inds.iter().enumerate() {
        positions[global] = i;
    }
    // Orthogonal Hermitian/Fourier coordinates use the Hilbert-Schmidt metric:
    // off-diagonal density entries count twice, as do the +/- Fourier pair.
    // This similarity transform makes the Hamiltonian part skew-adjoint.
    // Projecting the unweighted coordinate matrix can create spurious growing
    // Ritz modes merely because Re(rho_ij) and populations had different norms.
    let weights=fourier_weights(ext,model.dimension);
    let mut a = vec![vec![]; r];
    for (j, &global) in inds.iter().enumerate() {
        for &(row, v) in &ext.matrix[global] {
            if positions[row] != usize::MAX {
                a[j].push((positions[row], v * weights[row] / weights[global]));
            }
        }
    }
    let structured_product=if std::env::var("ATOM_STRUCTURED_FLOQUET_PRODUCT").is_ok_and(|value|value=="1"){
        match crate::structured_floquet::Prepared::new(model,structured_layout(ext,&inds,&weights)){
            Ok(prepared)=>{
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Structured full-density Floquet product ready: n={r}");}
                Some(std::rc::Rc::new(prepared))
            },
            Err(error)=>{
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Structured Floquet product unavailable; explicit matrix retained: {error}");}
                None
            },
        }
    }else{None};
    let mut initial: Vec<_> = inds
        .iter()
        .map(|&i| (ext.initial[i] - reference[i]) * weights[i])
        .collect();
    // Retain the COMPLETE defect of this reference, even if tiny. Long-time
    // weak pumping can accumulate forcing that appears negligible instantaneously.
    let forcing=if affine{
        let defect=sparse_product(&ext.matrix,reference);
        Some(inds.iter().map(|&i|defect[i]*weights[i]).collect::<Vec<_>>())
    }else{None};
    let mut trace_groups = vec![];
    for h in &ext.harmonics {
        for part in 0..if h.imaginary { 2 } else { 1 } {
            let rows: Vec<_> = (0..model.dimension)
                .filter_map(|i| {
                    let p = positions[h.offset + part * ext.physical_dimension + i];
                    (p != usize::MAX).then_some(p)
                })
                .collect();
            if !rows.is_empty() {
                trace_groups.push(rows);
            }
        }
    }
    let enforce_trace = |x: &mut [f64]| {
        for group in &trace_groups {
            let mean = group.iter().map(|&i| x[i]).sum::<f64>() / group.len() as f64;
            for &i in group {
                x[i] -= mean;
            }
        }
    };
    if traceless_transient {
        enforce_trace(&mut initial);
    }
    let times: Vec<_> = (0..options.samples)
        .map(|i| options.duration_us * i as f64 / (options.samples - 1) as f64)
        .collect();
    let physical = |vectors: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        vectors
            .into_iter()
            .zip(&times)
            .map(|(v, &t)| {
                let mut x = reference.to_vec();
                for (i, &global) in inds.iter().enumerate() {
                    x[global] += v[i] / weights[global];
                }
                let phases: Vec<_> = ext
                    .frequency_basis
                    .frequencies
                    .iter()
                    .map(|frequency| frequency * t)
                    .collect();
                reconstruct(ext, &x, &phases)
            })
            .collect()
    };
    if r <= 64
        || (!full_drive_mode() && r <= 512)
        || (!full_drive_mode() && std::env::var_os("ATOM_KRYLOV_EXPERIMENT").is_none())
        || std::env::var_os("ATOM_FLOQUET_DENSE").is_some()
    {
        if r > 8192 {
            return Err("The full Fourier propagator exceeds the current 8192-coordinate dense fallback budget; no state, carrier, or density coherence was removed.".into());
        }
        let mut dense = vec![0.; r * r];
        for (j, col) in a.iter().enumerate() {
            for &(i, v) in col {
                dense[i + j * r] = v;
            }
        }
        let evolved = if let Some(forcing)=&forcing{
            affine_spectral_action(lap,&dense,&initial,forcing,&times)?
        } else if r > 64 && max_abs(&dense) > 1e6 {
            refined_action(lap, &a, &initial, &times)?
        } else {
            spectral_action(lap, &dense, &initial, &times)?
        };
        let states = physical(evolved);
        return Ok(Trajectory {
            times,
            states,
            basis_order: r,
            convergence_error: 0.,
        });
    }
    let shift = std::env::var("ATOM_KRYLOV_SHIFT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|x| x.is_finite() && *x > 0.)
        .unwrap_or(if full_drive_mode() {
            // Resolve the requested output times. A fixed 10/us pole spends
            // most of a long-time solve building already-decayed fast modes.
            // This changes the numerical basis only; all density coordinates,
            // carriers, trace checks and nested convergence tests remain.
            ((options.samples - 1) as f64 / options.duration_us).clamp(1e-3, 10.)
        } else {
            1. / options.duration_us
        });
    let poles = if let Ok(experiment)=std::env::var("ATOM_KRYLOV_POLES_RAD"){
        absolute_numerical_poles(&experiment)?
    }else if let Ok(experiment)=std::env::var("ATOM_KRYLOV_POLES"){
        let omega=ext.frequency_basis.frequencies.iter().copied().fold(f64::INFINITY,f64::min);
        let selected:Vec<_>=experiment.split(',').filter_map(|s|s.trim().parse::<f64>().ok()).filter(|x|x.is_finite()&&*x>=0.).map(|x|x*omega).collect();if selected.is_empty(){vec![0.]}else{selected}
    }else if full_drive_mode() || std::env::var_os("ATOM_KRYLOV_SINGLE").is_some() {
        vec![0.]
    } else {
        numerical_poles(model, ext, shift)
    };
    let mut resolvents: Vec<Option<Resolvent>> = (0..poles.len()).map(|_| None).collect();
    if zero_seed.is_some()&&(poles.len()!=1||poles[0]!=0.){return Err("Constrained zero pole requires a single real numerical pole".into());}
    let beta = normv(&initial);
    if beta == 0. && forcing.as_ref().is_none_or(|b|b.iter().all(|&value|value==0.)) {
        return Ok(Trajectory {
            times: times.clone(),
            states: physical(vec![vec![0.; r]; times.len()]),
            basis_order: 0,
            convergence_error: 0.,
        });
    }
    let zero_resolvent=zero_seed.map(|mean|GaugeResolvent::new(lap,model,g,ext,&inds,&weights,&a,mean)).transpose()?;
    let mut factor_ms = zero_resolvent.as_ref().map(|gauge|gauge.setup_ms).unwrap_or(0.);
    let mut inverse_ms = 0.;
    let mut orthogonal_ms = 0.;
    let mut projection_ms = 0.;
    let mut basis = if beta>0.{vec![initial.iter().map(|x| x / beta).collect::<Vec<_>>()]}else{vec![]};
    if let Some(forcing)=&forcing{
        let forcing_norm=normv(forcing);
        if forcing_norm>0.{
            // Normalize BEFORE orthogonalizing: small but nonzero forcing
            // receives its own successor chain on the same footing as δ0.
            let mut v:Vec<_>=forcing.iter().map(|x|x/forcing_norm).collect();
            if traceless_transient{enforce_trace(&mut v);}
            for _ in 0..2{for q in &basis{let dot=dotv(q,&v);for(x,&y)in v.iter_mut().zip(q){*x-=dot*y;}}}
            let length=normv(&v);
            if length>1e-12{basis.push(v.into_iter().map(|x|x/length).collect());}
        }
    }
    if !traceless_transient {
        for group in &trace_groups {
            let mut v = vec![0.; r];
            for &i in group {
                v[i] = 1.;
            }
            for _ in 0..2 {
                for q in &basis {
                    let dot = dotv(q, &v);
                    for (i, x) in v.iter_mut().enumerate() {
                        *x -= dot * q[i];
                    }
                }
            }
            let norm = normv(&v);
            if norm > 1e-12 {
                basis.push(v.into_iter().map(|x| x / norm).collect());
            }
        }
    }
    if full_drive_mode() && ext.frequency_basis.frequencies.len() == 1 && poles.len()==1 && std::env::var_os("ATOM_KRYLOV_NO_RECYCLE").is_none() {
        RECYCLED_SPACE.with(|cache| {
            if let Some(old) = cache.borrow().as_ref() {
                for col in old.columns.iter().skip(1).take(47) {
                    let mut v = vec![0.; r];
                    for (i, &global) in old.indices.iter().enumerate() {
                        if global < positions.len() && positions[global] != usize::MAX {
                            v[positions[global]] = col[i];
                        }
                    }
                    enforce_trace(&mut v);
                    for _ in 0..2 {
                        for q in &basis {
                            let dot = dotv(q, &v);
                            for (i, x) in v.iter_mut().enumerate() {
                                *x -= dot * q[i];
                            }
                        }
                    }
                    let length = normv(&v);
                    if length > 1e-12 {
                        basis.push(v.into_iter().map(|v| v / length).collect());
                    }
                }
            }
        });
    }
    let seeded_size = basis.len();
    if seeded_size==0{return Err("Affine forcing has no numerically resolved trace-free seed".into());}
    let cyclic_poles=std::env::var("ATOM_KRYLOV_CYCLIC_POLES").is_ok_and(|value|value=="1");
    if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Krylov numerical poles: realShift={shift:e} frequencies={poles:?} cyclicGeneration={cyclic_poles}");}
    // Every independent seed needs its own successor chain. In particular,
    // recycling must never leave the current initial-density seed unexpanded.
    let mut frontier = (seeded_size>1||poles.len()>1||std::env::var("ATOM_KRYLOV_FRONTIER").is_ok_and(|value|value=="1"))
        .then(|| KrylovFrontier::with_cyclic_poles(seeded_size, poles.len(),cyclic_poles));
    let mut previous: Option<Vec<Vec<f64>>> = None;
    let limit = r.min(768);
    let mut best_delta = f64::INFINITY;
    let mut checked_size:usize = 0;
    let mut unstable_checks = 0;
    // A long sampled affine transient need not build eight more directions
    // AFTER satisfying all density,
    // population, complete-generation, trace and PSD checks. Harmonic
    // refinement still independently repeats the whole trajectory above.
    let default_minimum=if affine&&full_drive_mode()&&prefer_affine_center(model.dimension,ext.frequency_basis.frequencies.len(),options.duration_us,options.samples,options.steady_only){24}else{32};
    let minimum_basis = std::env::var("ATOM_KRYLOV_MIN_BASIS").ok().and_then(|s|s.parse::<usize>().ok()).filter(|n|*n>=16).unwrap_or(default_minimum);
    let instability_floor=if poles.len()>1{128}else{32};
    let mut basis_products = Vec::new();
    let incremental_projection=std::env::var("ATOM_INCREMENTAL_PROJECTED").is_ok_and(|v|v=="1");
    let mut projected_prefix=crate::incremental_projection::Projection::new();
    let append = |basis: &mut Vec<Vec<f64>>, frontier: &mut Option<KrylovFrontier>, additions: Vec<Vec<f64>>| {
        for vector in additions.into_iter().take(limit-basis.len()) {
            let index=basis.len();basis.push(vector);
            if let Some(work)=frontier {work.append(index);}
        }
    };
    for attempt in 0..limit * poles.len().max(2) {
        let mut size = basis.len();
        let (source,pole) = if let Some(work)=&mut frontier {
            match work.next() {Some(job)=>job,None=>break}
        } else {(size-1,attempt%poles.len())};
        let factor_started = Instant::now();
        if zero_resolvent.is_none()&&resolvents[pole].is_none() {
            let custom:Option<crate::block_solve::CustomApply>=if model.dimension==32&&poles[pole]==0.&&std::env::var("ATOM_OPTICAL_SCHUR").is_ok_and(|value|value=="1"){
                let mut original:Sparse=a.iter().map(|column|column.iter().map(|&(i,value)|(i,-value)).collect()).collect();
                for(i,column)in original.iter_mut().enumerate(){column.push((i,shift));}compress(&mut original);
                let ground:Vec<_>=(0..model.dimension).map(|i|model.states.get(i).is_some_and(|id|id.starts_with("ground-"))).collect();
                let qmask:Vec<_>=inds.iter().map(|&global|match g.coords[global%ext.physical_dimension]{
                    Coordinate::Population(_)=>false,Coordinate::Real(i,j)|Coordinate::Imag(i,j)=>ground[i]!=ground[j],
                }).collect();
                let prepared=if std::env::var("ATOM_OPTICAL_FEEDBACK").is_ok_and(|value|value=="1"){
                    let mut groups=vec![0;ext.matrix.len()];
                    for(group,h)in ext.harmonics.iter().enumerate(){
                        groups[h.offset..h.offset+ext.physical_dimension*if h.imaginary{2}else{1}].fill(group);
                    }
                    let groups:Vec<_>=inds.iter().map(|&i|groups[i]).collect();
                    crate::optical_schur::Prepared::new_with_groups(lap,&original,&qmask,&groups)
                }else{crate::optical_schur::Prepared::new(lap,&original,&qmask)};
                match prepared{
                    Ok(mut prepared)=>{
                        if let Some(product)=&structured_product{
                            let product=std::rc::Rc::clone(product);
                            prepared=prepared.with_raw_product(Box::new(move|x|product.product_shifted(x,shift,None)));
                        }
                        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Exact optical Schur prepared: {:?}",prepared.diagnostics());}
                        let retired=Cell::new(false);let fallback=RefCell::new(None::<crate::block_solve::Prepared>);
                        let product=structured_product.as_ref().map(std::rc::Rc::clone);
                        Some(Box::new(move|lap,rhs|{
                            if !retired.get(){match prepared.solve(lap,rhs){
                                Ok(value)=>return Ok(value),
                                Err(error)=>{retired.set(true);if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Optical Schur retired; restore original full ILUTP: {error}");}},
                            }}
                            if fallback.borrow().is_none(){
                                let mut full=crate::block_solve::Prepared::new(lap,&original,&[],0.)?;
                                if let Some(product)=&product{let product=std::rc::Rc::clone(product);full=full.with_raw_product(Box::new(move|x|product.product_shifted(x,shift,None)));}
                                *fallback.borrow_mut()=Some(full);
                            }
                            fallback.borrow().as_ref().unwrap().solve(lap,rhs)
                        }))
                    },
                    Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Optical Schur unavailable; original route retained: {error}");}None},
                }
            }else if std::env::var_os("ATOM_SYLVESTER_PRECONDITIONER").is_some(){
                let blocks:Vec<_>=ext.harmonics.iter().map(|h|(h.offset,h.index.iter().zip(&ext.frequency_basis.frequencies).map(|(&k,w)|k as f64*w).sum(),h.imaginary)).collect();
                if poles[pole]!=0.{match crate::sylvester_recycling::ComplexHarmonicBlocks::new(lap,model,shift,poles[pole],&blocks,ext.physical_dimension,&inds,&weights){
                    Ok(prepared)=>Some(Box::new(move|lap,rhs|prepared.apply_preconditioner(lap,rhs))),
                    Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Complex Sylvester preconditioner preparation fallback: {error}");}None}
                }}else{match crate::sylvester_recycling::HarmonicBlocks::new(lap,model,shift,&blocks,ext.physical_dimension,&inds,&weights){
                    Ok(prepared)=>Some(Box::new(move|lap,rhs|prepared.apply_preconditioner(lap,rhs))),
                    Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Sylvester preconditioner preparation fallback: {error}");}None}
                }}
            }else{None};
            resolvents[pole] = Some(Resolvent::new(
                lap,
                &a,
                shift,
                poles[pole],
                &inds
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &g)| (g < ext.physical_dimension).then_some(i))
                    .collect::<Vec<_>>(),
                ext.frequency_basis
                    .frequencies
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
                custom,
                &inds.iter().enumerate().filter_map(|(local,&global)|{
                    if global>=ext.physical_dimension{return None;}
                    let slow=match g.coords[global]{Coordinate::Population(i)=>i<8,Coordinate::Real(i,j)|Coordinate::Imag(i,j)=>i<8&&j<8&&(i<3)==(j<3)};
                    slow.then_some(local)
                }).collect::<Vec<_>>(),
                if full_drive_mode()&&poles[pole]==0.&&std::env::var("ATOM_REUSE_HARMONIC_ILUTP").is_ok_and(|value|value=="1"){
                    Some(refinement_layout(ext,&inds,&weights)?)
                }else{None},
            )?);
            if let Some(prepared)=&structured_product{
                let resolvent=resolvents[pole].as_mut().unwrap();
                if let Some(block)=resolvent.block.take(){
                    let prepared=std::rc::Rc::clone(prepared);
                    let external_frequency=resolvent.augmented.then_some(poles[pole]);
                    resolvent.block=Some(block.with_raw_product(Box::new(move|x|prepared.product_shifted(x,shift,external_frequency))));
                }
            }
        }
        factor_ms += factor_started.elapsed().as_secs_f64() * 1000.;
        let inverse_started = Instant::now();
        let (real, imag) = if let Some(gauge)=&zero_resolvent{(gauge.apply(lap,&a,&basis[source])?,vec![0.;r])}else{
            resolvents[pole].as_ref().unwrap().apply(lap,&basis[source])?
        };
        inverse_ms += inverse_started.elapsed().as_secs_f64() * 1000.;
        let orthogonal_started = Instant::now();
        let original_norm = normv(&real).hypot(normv(&imag));
        let mut candidates = [real, imag];
        let mut additions:Vec<Vec<f64>>=Vec::new();
        // Two-pass modified Gram-Schmidt controls loss of orthogonality when
        // the carrier and relaxation frequencies differ by many orders.
        for w in &mut candidates {
            if w.iter().all(|v|*v==0.) {continue;}
            if traceless_transient {
                enforce_trace(w);
            }
            for _ in 0..2 {
                for v in &basis {
                    let projection = dotv(v, w);
                    for (i, wi) in w.iter_mut().enumerate() {
                        *wi -= projection * v[i];
                    }
                }
                if frontier.is_some() {
                    for v in &additions {
                        let projection=dotv(v,w);
                        for(i,wi)in w.iter_mut().enumerate(){*wi-=projection*v[i];}
                    }
                }
            }
            enforce_trace(w);
            if frontier.is_some() {
                let length=normv(w);
                if length>=1e-12*original_norm && length>0. {
                    additions.push(w.iter().map(|x|x/length).collect());
                }
            }
        }
        let [real, imag] = candidates;
        let w = if normv(&real) >= normv(&imag) {
            real
        } else {
            imag
        };
        let length = normv(&w);
        let exhausted = if frontier.is_some(){additions.is_empty()}else{length < 1e-12 * original_norm};
        if frontier.is_none()&&!exhausted{additions.push(w.into_iter().map(|x|x/length).collect());}
        orthogonal_ms += orthogonal_started.elapsed().as_secs_f64() * 1000.;
        // A multi-start convergence comparison must follow a COMPLETE block
        // generation, not several unrelated recycled children. Otherwise a
        // still-unexpanded δ0/b descendant can be hidden by a false plateau.
        let completed_generation=frontier.as_ref().is_some_and(KrylovFrontier::generation_complete);
        if completed_generation{
            append(&mut basis,&mut frontier,std::mem::take(&mut additions));size=basis.len();
            frontier.as_mut().unwrap().begin_next_generation();
        }
        let closed_frontier=completed_generation&&frontier.as_ref().is_some_and(|work|work.jobs.is_empty());
        let check_projection=if frontier.is_some(){
            // Complete generations remain mandatory. Reconstructing every
            // sampled density after each two-vector generation is redundant;
            // use the same bounded check spacing as the single-start path.
            let interval=if full_drive_mode(){if size<32{8}else{16}}else{32};
            frontier_projection_due(size,checked_size,interval,completed_generation,closed_frontier,limit)
        }else{
            size % (if full_drive_mode() { if minimum_basis<32 && size<32 {8} else {16} } else {32}) == 0
            || size == limit
            || (seeded_size > 1 && (size == seeded_size || size == seeded_size + 4))
        };
        if check_projection && (checked_size != size||closed_frontier)
        {
            checked_size = size;
            let projection_started = Instant::now();
            let columns: Vec<f64> = basis.iter().flatten().copied().collect();
            // Existing basis vectors never change. Apply A only to newly
            // added columns, rather than repeating every earlier product.
            while basis_products.len()<size*r {
                let index=basis_products.len()/r;
                if std::env::var("ATOM_STRUCTURED_OUTER_PRODUCT").is_ok_and(|value|value=="1") {
                    if let Some(prepared)=&structured_product {
                        // Same complete generator as the explicit matrix:
                        // product_shifted(x,0) = -A*x. No coordinate is dropped.
                        basis_products.extend(prepared.try_product_shifted(&basis[index],0.,None)?.into_iter().map(|v|-v));
                    } else {basis_products.extend(sparse_product(&a,&basis[index]));}
                } else {basis_products.extend(sparse_product(&a,&basis[index]));}
            }
            // Q and A are immutable within this trajectory; retain the old
            // projected block and compute only newly appended rows/columns.
            let projected=if incremental_projection{
                projected_prefix.update(lap,&columns,&basis_products,r,size)?
            }else{
                let transpose:Vec<f64>=(0..r).flat_map(|k|basis.iter().map(move|v|v[k])).collect();
                lap.multiply(&transpose,&basis_products,size,size,r)
            };
            if closed_frontier&&zero_resolvent.as_ref().is_some_and(|gauge|!gauge.closed_space_is_invariant(&basis,&basis_products,&projected)){
                return Err("Zero-pole numerical stagnation: the closed inverse space is not invariant under the complete generator".into());
            }
            let start:Vec<_>=basis.iter().map(|q|dotv(q,&initial)).collect();
            let action=if let Some(forcing)=&forcing{
                let projected_forcing:Vec<_>=basis.iter().map(|q|dotv(q,forcing)).collect();
                affine_spectral_action(lap,&projected,&start,&projected_forcing,&times)
            }else{spectral_action(lap,&projected,&start,&times)};
            let small = match action{Ok(rows)=>rows,Err(error)=>{
                // A coarse projected space may have unstable Ritz modes even
                // though the original Lindblad evolution is bounded. Refine
                // that numerical space instead of aborting a long-time request.
                unstable_checks+=1;
                if full_drive_mode()&&unstable_checks>=2&&size>=instability_floor{return Err(format!("Unstable projected exponential at basis {size}: {error}"));}
                if size>=limit{return Err(error);}previous=None;append(&mut basis,&mut frontier,additions);continue;
            }};
            let sample_count = small.len();
            let weights: Vec<f64> = small.into_iter().flatten().collect();
            let lifted = lap.multiply(&columns, &weights, r, sample_count, size);
            let vectors = lifted.chunks(r).map(|row| row.to_vec()).collect();
            let states = physical(vectors);
            projection_ms += projection_started.elapsed().as_secs_f64() * 1000.;
            if full_drive_mode(){
                let unsafe_growth=states.iter().any(|x|x.iter().any(|v|!v.is_finite()||v.abs()>2.)||x[..model.dimension].iter().any(|v|*v< -0.05||*v>1.05));
                if unsafe_growth {unstable_checks+=1;}else{unstable_checks=0;}
                if unstable_checks>=2&&size>=instability_floor{return Err(format!("Unstable projected exponential at basis {size}; abandon this Fourier order instead of amplifying a nonphysical mode"));}
            }
            if let Some(old) = &previous {
                best_delta = states
                    .iter()
                    .zip(old)
                    .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
                    .fold(0., f64::max);
            }
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                let pop_delta = previous.as_ref().map(|old| {
                    states
                        .iter()
                        .zip(old)
                        .flat_map(|(a, b)| {
                            a[..model.dimension]
                                .iter()
                                .zip(&b[..model.dimension])
                                .map(|(a, b)| (a - b).abs())
                        })
                        .fold(0., f64::max)
                });
                let trace = states
                    .iter()
                    .map(|x| (x[..model.dimension].iter().sum::<f64>() - 1.).abs())
                    .fold(0., f64::max);
                eprintln!("Phases factor={factor_ms:.2} inverse={inverse_ms:.2} orthogonal={orthogonal_ms:.2} projection={projection_ms:.2} ms");
                eprintln!("Krylov {size}/{r}: delta={best_delta:e}, pop={pop_delta:?}, trace={trace:e}, innovation={length:e}");
            }
            let population_delta = previous
                .as_ref()
                .map(|old| {
                    states
                        .iter()
                        .zip(old)
                        .flat_map(|(a, b)| {
                            a[..model.dimension]
                                .iter()
                                .zip(&b[..model.dimension])
                                .map(|(a, b)| (a - b).abs())
                        })
                        .fold(0., f64::max)
                })
                .unwrap_or(f64::INFINITY);
            if (if full_drive_mode() {
                best_delta < 1e-3 && population_delta < 5e-5
            } else {
                best_delta < 1e-8
            }) && size >= minimum_basis
                && frontier.as_ref().is_none_or(KrylovFrontier::seeds_expanded)
                && (frontier.is_none()||completed_generation)
            {
                // Population agreement alone can hide inaccurate coherences.
                // Reject the projected candidate here while its basis can
                // still grow, rather than rediscovering the same invalid
                // density at the next Fourier order and failing at the end.
                let (trace_error,positivity_error)=trajectory_density_error(lap,g,&states)?;
                if trace_error<=1e-7&&positivity_error<=if full_drive_mode(){1e-4}else{1e-7} {
                if full_drive_mode() {
                    RECYCLED_SPACE.with(|cache| {
                        *cache.borrow_mut() = Some(RecycledSpace {
                            indices: inds.clone(),
                            columns: basis.clone(),
                        })
                    });
                }
                return Ok(Trajectory {
                    times,
                    states,
                    basis_order: size,
                    convergence_error: if full_drive_mode() {
                        population_delta
                    } else {
                        best_delta
                    },
                });
                } else if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                    eprintln!("Krylov {size}/{r}: density candidate rejected: trace={trace_error:e}, positivity={positivity_error:e}; expand the same basis");
                }
            }
            previous = Some(states);
        }
        if size == limit {
            break;
        }
        if exhausted {
            continue;
        }
        append(&mut basis,&mut frontier,additions);
    }
    if let (Some(path), Some(states)) = (std::env::var_os("ATOM_KRYLOV_AUDIT"), previous) {
        let _=std::fs::write(path,json!({"accepted":false,"shift":if zero_resolvent.is_some(){0.}else{shift},"dimension":r,"timesUs":times,"coordinates":states,"convergenceDelta":best_delta}).to_string());
    }
    Err(format!("Full Fourier matrix-exponential Krylov space did not converge (last density change {best_delta:e}); no approximate trajectory returned"))
}
fn lift_reference(old:&Extended,coefficients:&[f64],new:&Extended)->Result<Vec<f64>,String>{
    if old.physical_dimension!=new.physical_dimension||old.frequency_basis.frequencies!=new.frequency_basis.frequencies
        ||old.frequency_basis.drive_indices!=new.frequency_basis.drive_indices||coefficients.len()!=old.matrix.len(){
        return Err("Cannot lift a Fourier reference across different physical coordinates or frequency bases".into());
    }
    let positions:HashMap<_,_>=new.harmonics.iter().map(|h|(h.index.as_slice(),h)).collect();
    let mut lifted=vec![0.;new.matrix.len()];
    for h in &old.harmonics{
        let target=positions.get(h.index.as_slice()).ok_or("Refined harmonic lattice omitted a reference harmonic")?;
        if target.imaginary!=h.imaginary{return Err("Refined Fourier reference changed harmonic representation".into());}
        let width=old.physical_dimension*if h.imaginary{2}else{1};
        lifted[target.offset..target.offset+width].copy_from_slice(&coefficients[h.offset..h.offset+width]);
    }
    Ok(lifted)
}
struct AffineRefinement{
    trajectory:Trajectory,order:i32,dimension:usize,delta:f64,failures:Vec<Value>,restarts:usize,
}
fn prefer_affine_center(dimension:usize,axes:usize,duration_us:f64,samples:usize,steady_only:bool)->bool{
    !steady_only&&dimension==32&&axes==1&&samples>=2&&duration_us.is_finite()
        &&duration_us/(samples-1)as f64>10.
}
fn affine_center_requested(model:&Model,ext:&Extended,options:&Options)->bool{
    match std::env::var("ATOM_AFFINE_CENTER").as_deref(){
        Ok("1")=>true,
        Ok("auto")=>prefer_affine_center(model.dimension,ext.frequency_basis.frequencies.len(),options.duration_us,options.samples,options.steady_only),
        _=>false,
    }
}
/// Once the mean has independently converged, its coefficients are only a
/// numerical origin for finite propagation. Every larger A retains A*c as
/// constant forcing, so no additional stationary equation needs to be solved.
fn affine_refine_trajectory(lap:&Lapack,model:&Model,g:&Generator,mean:&Stationary,mean_order:i32,options:&Options)->Result<AffineRefinement,String>{
    if !mean.valid||!mean.condition.is_finite(){return Err("Affine refinement requires a separately certified mean".into());}
    let mut previous:Option<Trajectory>=None;let mut failures=vec![];let mut restarts=0;
    for order in mean_order..=12{
        let refined=if order==mean_order{None}else{Some(extend(model,g,order)?)};
        let ext=refined.as_ref().unwrap_or(&mean.extended);
        let reference=lift_reference(&mean.extended,&mean.coefficients,ext)?;
        let had_seed=RECYCLED_SPACE.with(|space|space.borrow().is_some());
        let run=||trajectory_with_mean(lap,model,g,ext,&reference,false,true,options,mean);
        let mut result=run();
        if had_seed&&result.as_ref().err().is_some_and(|e|e.contains("Unstable projected exponential")){
            RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);restarts+=1;result=run();
        }
        let current=match result{
            Ok(value)=>value,
            Err(error)if error.contains("Unstable projected exponential")||error.contains("Krylov space did not converge")=>{
                failures.push(json!({"order":order,"reason":error}));
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Affine trajectory K={order}: rejected: {error}");}
                if failures.len()>=3{return Err(format!("Affine time evolution failed across three Fourier refinements: {error}"));}
                RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);previous=None;continue;
            },
            Err(error)=>return Err(error),
        };
        let delta=previous.as_ref().map(|old|current.states.iter().zip(&old.states)
            .flat_map(|(a,b)|a[..model.dimension].iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max)).unwrap_or(f64::INFINITY);
        let(trace_error,positivity_error)=trajectory_density_error(lap,g,&current.states)?;
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Affine trajectory K={order} (mean K={mean_order}): population refinement={delta:e}, trace={trace_error:e}, positivity={positivity_error:e}");}
        if delta<5e-5&&trace_error<=1e-7&&positivity_error<=1e-4{
            let dimension=reachable(&ext.matrix,&ext.initial).len();
            return Ok(AffineRefinement{trajectory:current,order,dimension,delta,failures,restarts});
        }
        previous=Some(current);
    }
    Err("Affine Fourier trajectory refinement did not converge within order 12; no trajectory returned".into())
}

fn independent_finite_groups(model:&Model)->Vec<Vec<usize>>{
    if model.states.len()!=model.dimension{return Vec::new();}
    let mut groups=std::collections::BTreeMap::<String,Vec<usize>>::new();
    for(i,state)in model.states.iter().enumerate(){
        groups.entry(state.split('-').take(2).collect::<Vec<_>>().join("-")).or_default().push(i);
    }
    groups.into_values().collect()
}
fn independent_finite_retryable(error:&str)->bool{
    // Only these numerical projection/refinement failures can improve with a
    // larger trial space. Input, metadata, factor, and protocol errors escape.
    error.contains("Unstable projected exponential")||error.contains("Krylov space did not converge")
}
/// Finite time does not require an infinite-time density. On EVERY retained
/// complete Fourier operator A, c=initial and delta'=A*delta+A*c is an exact
/// identity. The full forcing is retained even for nonunique/closed systems.
/// This entry deliberately calls the positive-pole path with zero_seed=None;
/// it never prepares a stationary equation, inverse, or condition estimate.
fn independent_finite_result(lap:&Lapack,model:&Model,g:&Generator,options:&Options)->Result<Value,String>{
    if options.steady_only||options.include_asymptotic||model.drives.is_empty()
        ||options.samples<2||options.samples>100000||!options.duration_us.is_finite()||options.duration_us<=0.{
        return Err("Independent finite propagation requires a driven finite-only request and a valid output grid".into());
    }
    let started=Instant::now();let groups=independent_finite_groups(model);
    let mut previous:Option<(i32,usize,Trajectory)>=None;let mut attempts=Vec::new();let mut failures=Vec::new();let mut restarts=0;
    let mut last_error=String::from("No consecutive finite trajectories were available");
    for order in 1..=12{
        let stage=Instant::now();let ext=extend(model,g,order)?;let extended_dimension=ext.matrix.len();
        let had_seed=RECYCLED_SPACE.with(|space|space.borrow().is_some());
        let run=||trajectory_from_reference(lap,model,g,&ext,&ext.initial,false,true,options);
        let mut outcome=run();
        if had_seed&&outcome.as_ref().err().is_some_and(|error|error.contains("Unstable projected exponential")){
            RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);restarts+=1;outcome=run();
        }
        let current=match outcome{
            Ok(value)=>value,
            Err(error)if independent_finite_retryable(&error)=>{
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Independent finite K={order}: rejected: {error}");}
                failures.push(json!({"order":order,"reason":error}));
                attempts.push(json!({"order":order,"extendedDimension":extended_dimension,"elapsedMs":stage.elapsed().as_secs_f64()*1000.,"error":error}));
                last_error=error;previous=None;RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);continue;
            },
            Err(error)=>return Err(error),
        };
        if current.times.len()!=options.samples||current.states.len()!=options.samples||current.states.iter().any(|state|state.len()!=g.n*g.n){
            return Err("Independent finite trajectory returned inconsistent dimensions".into());
        }
        let(trace_error,positivity_error)=trajectory_density_error(lap,g,&current.states)?;
        let initial_error=current.states[0].iter().zip(&g.initial).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
        if trace_error>1e-7||positivity_error>1e-4||!initial_error.is_finite()||initial_error>1e-10{
            let error=format!("Finite density requires refinement: trace={trace_error:e}, positivity={positivity_error:e}, initial={initial_error:e}");
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Independent finite K={order}: {error}");}
            failures.push(json!({"order":order,"reason":error}));
            attempts.push(json!({"order":order,"extendedDimension":extended_dimension,"elapsedMs":stage.elapsed().as_secs_f64()*1000.,"traceError":trace_error,"positivityError":positivity_error,"initialStateError":initial_error,"error":error}));
            last_error=error;previous=None;RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);continue;
        }
        let mut population_delta=f64::INFINITY;let mut group_delta=f64::INFINITY;let mut density_delta=f64::INFINITY;
        if let Some((previous_order,previous_dimension,old))=&previous{
            if *previous_order+1!=order||*previous_dimension>=extended_dimension||old.times!=current.times{
                return Err("Independent finite comparison requires consecutive larger Fourier spaces on the same output grid".into());
            }
            population_delta=0.;group_delta=0.;density_delta=0.;
            for(a,b)in current.states.iter().zip(&old.states){
                for i in 0..g.n{population_delta=population_delta.max((a[i]-b[i]).abs());}
                for group in &groups{group_delta=group_delta.max(group.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
                for(a,b)in a.iter().zip(b){density_delta=density_delta.max((a-b).abs());}
            }
        }
        attempts.push(json!({"order":order,"extendedDimension":extended_dimension,"elapsedMs":stage.elapsed().as_secs_f64()*1000.,
            "krylovDimension":current.basis_order,"krylovConvergenceError":current.convergence_error,"populationDifference":population_delta,"groupPopulationDifference":group_delta,
            "densityDifference":density_delta,"traceError":trace_error,"positivityError":positivity_error,"initialStateError":initial_error}));
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Independent finite K={order}: population={population_delta:e}, Fgroup={group_delta:e}, density={density_delta:e}, trace={trace_error:e}, positivity={positivity_error:e}; no stationary solve");}
        if population_delta<5e-5&&group_delta<5e-5&&density_delta<1e-3{
            let rows:Vec<_>=current.states.iter().map(|state|expand_density(&g.coords,state,g.n)).collect();
            let density=options.return_density.then_some(&rows);
            return Ok(json!({"timesUs":current.times,"states":if model.states.len()==g.n{model.states.clone()}else{(0..g.n).map(|i|i.to_string()).collect::<Vec<_>>()},
                "populations":current.states.iter().map(|state|state[..g.n].to_vec()).collect::<Vec<_>>(),
                "densityReal":density.map(|rows|rows.iter().map(|row|row.iter().map(|z|z.re).collect::<Vec<_>>()).collect::<Vec<_>>()),
                "densityImag":density.map(|rows|rows.iter().map(|row|row.iter().map(|z|z.im).collect::<Vec<_>>()).collect::<Vec<_>>()),
                "scatteringRatePerUs":rows.iter().map(|rho|scatter_rate(&g.scattering,rho,g.n)).collect::<Vec<_>>(),
                "asymptoticPopulations":null,"asymptoticDensityReal":null,"asymptoticDensityImag":null,"asymptoticFourierDensity":null,"asymptoticStatus":"not-requested",
                "diagnostics":{"engine":"rust-floquet","method":"Complete Fourier finite evolution with exact initial-state affine forcing",
                    "independentFinite":true,"finiteReference":"initial-state","stationarySolveMs":0.,"stationaryOnly":false,"stationarySolved":false,
                    "liouvilleDimension":g.n*g.n,"reachableDimension":reachable(&ext.matrix,&ext.initial).len(),"extendedDimension":extended_dimension,
                    "harmonicOrder":order,"harmonicConvergenceError":population_delta.max(group_delta),"populationHarmonicConvergenceError":population_delta,
                    "groupPopulationHarmonicConvergenceError":group_delta,"densityHarmonicConvergenceError":density_delta,
                    "krylovDimension":current.basis_order,"krylovConvergenceError":current.convergence_error,"trajectoryBasisRestarts":restarts,"trajectoryRejectedOrders":failures,
                    "finiteRefinementAttempts":attempts,"independentBeats":ext.frequency_basis.frequencies.len(),"frequencyBasis":ext.frequency_basis.frequencies,
                    "frequencyReconstructionErrors":ext.frequency_basis.reconstruction_errors,"roundedFrequencyRelations":ext.frequency_basis.rounded_relations,
                    "traceError":trace_error,"positivityError":positivity_error,"hermiticityError":0.,"initialStateError":initial_error,
                    "elapsedMs":started.elapsed().as_secs_f64()*1000.,"asymptoticStatus":"not-requested",
                    "errorScope":"Consecutive larger Fourier spaces at every requested time; full affine forcing, population/F-group/full-density refinement and unchanged trace/PSD checks. No stationary equation or infinite-time conclusion."}}));
        }
        last_error=format!("population={population_delta:e}, Fgroup={group_delta:e}, density={density_delta:e}");
        previous=Some((order,extended_dimension,current));
    }
    Err(format!("Independent finite Fourier refinement did not converge within order 12: {last_error}; no trajectory returned"))
}

#[cfg(test)]
mod independent_finite_tests{
    use super::*;
    fn run(lap:&Lapack,model:&Model,options:&Options)->Value{
        clear_request_results();RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);
        STATIONARY_PROFILE.with(|profile|*profile.borrow_mut()=Some(Vec::new()));
        let old=FULL_DRIVE_MODE.with(|flag|flag.replace(true));let g=build_generator(model).unwrap();
        let result=independent_finite_result(lap,model,&g,options);
        FULL_DRIVE_MODE.with(|flag|flag.set(old));RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);
        REFINEMENT_PRECONDITIONER.with(|factor|*factor.borrow_mut()=None);
        assert_eq!(ZERO_POLE_REPORT.with(|report|report.borrow().attempts),0);
        assert!(STATIONARY_PROFILE.with(|profile|profile.borrow().as_ref().unwrap().is_empty()));
        let result=result.unwrap();assert_eq!(result["diagnostics"]["stationarySolved"],false);
        assert_eq!(result["diagnostics"]["stationarySolveMs"],0.);assert_eq!(result["asymptoticStatus"],"not-requested");
        assert!(result["asymptoticPopulations"].is_null());assert!(result["diagnostics"]["harmonicOrder"].as_i64().unwrap()>=2);result
    }
    fn density(result:&Value,sample:usize)->Vec<C>{
        result["densityReal"][sample].as_array().unwrap().iter().zip(result["densityImag"][sample].as_array().unwrap())
            .map(|(re,im)|C::new(re.as_f64().unwrap(),im.as_f64().unwrap())).collect()
    }
    fn options()->Options{Options{duration_us:2.,samples:13,include_asymptotic:false,return_density:true,..Options::default()}}
    #[test]
    fn independent_finite_closed_unitary_matches_analytic_rabi_without_a_mean(){
        let lap=Lapack::load().unwrap();let model=Model{dimension:2,states:vec![],h_real:vec![0.,0.3,0.3,0.],h_imag:vec![0.;4],collapse:vec![],
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:true,frequency_coordinates:None,
            drives:vec![Drive{omega:2.3,cos:Operator{real:vec![0.,0.04,0.04,0.],imag:vec![0.;4]},sin:Operator{real:vec![0.,-0.02,-0.02,0.],imag:vec![0.;4]}}]};
        let result=run(&lap,&model,&options());
        for(sample,time)in result["timesUs"].as_array().unwrap().iter().enumerate(){let t=time.as_f64().unwrap();
            let angle=0.3*t+0.04*(2.3*t).sin()/2.3-0.02*(1.-(2.3*t).cos())/2.3;
            let actual=density(&result,sample);let(s,c)=angle.sin_cos();let expected=[C::new(c*c,0.),C::new(0.,c*s),C::new(0.,-c*s),C::new(s*s,0.)];
            assert!(actual.iter().zip(expected).all(|(&a,b)|(a-b).abs()<2e-5),"closed finite density at t={t}");
        }
    }
    #[test]
    fn independent_finite_nonunique_spectator_retains_each_initial_mixture(){
        let lap=Lapack::load().unwrap();let mut model=Model{dimension:3,states:vec!["ground-2-0".into(),"ground-2-1".into(),"ground-1-0".into()],
            h_real:vec![0.,0.2,0.,0.2,0.7,0.,0.,0.,0.],h_imag:vec![0.;9],collapse:vec![Operator{real:vec![0.,0.6,0.,0.,0.,0.,0.,0.,0.],imag:vec![0.;9]}],
            initial_index:0,initial_real:None,initial_imag:Some(vec![0.;9]),time_dependent:true,frequency_coordinates:None,
            drives:vec![Drive{omega:2.3,cos:Operator{real:vec![0.,0.03,0.,0.03,0.,0.,0.,0.,0.],imag:vec![0.;9]},
                sin:Operator{real:vec![0.;9],imag:vec![0.,0.015,0.,-0.015,0.,0.,0.,0.,0.]}}]};
        let options=Options{duration_us:0.7,samples:9,..options()};let mut final_populations=Vec::new();
        for weight in [0.2_f64,0.7]{
            model.initial_real=Some(vec![weight,0.,0.,0.,0.,0.,0.,0.,1.-weight]);
            let result=run(&lap,&model,&options);let g=build_generator(&model).unwrap();let ext=extend(&model,&g,4).unwrap();let size=ext.matrix.len();
            let mut a=vec![0.;size*size];for(j,column)in ext.matrix.iter().enumerate(){for &(i,value)in column{a[i+j*size]=value;}}
            let times:Vec<_>=result["timesUs"].as_array().unwrap().iter().map(|value|value.as_f64().unwrap()).collect();
            // Independent direct exponential of the full finite generator,
            // with neither an affine center nor a stationary solve.
            let expected=crate::dense_expm::action(&lap,&a,&ext.initial,&times).unwrap();
            for(sample,(coefficients,&time))in expected.iter().zip(&times).enumerate(){
                let coordinates=reconstruct(&ext,coefficients,&[2.3*time]);let reference=expand_density(&g.coords,&coordinates,3);let actual=density(&result,sample);
                assert!(actual.iter().zip(reference).all(|(&a,b)|(a-b).abs()<2e-5));
                assert!((actual[8].re-(1.-weight)).abs()<2e-13);assert!((actual[0].re+actual[4].re-weight).abs()<2e-13);
            }
            final_populations.push(result["populations"][8][1].as_f64().unwrap());
            assert!(result["diagnostics"]["groupPopulationHarmonicConvergenceError"].as_f64().unwrap()<5e-5);
        }
        assert!(final_populations[0]>1e-5);assert!((final_populations[1]/final_populations[0]-3.5).abs()<2e-6);
    }
    #[test]
    fn independent_finite_zero_forcing_returns_initial_and_rejects_nonfinite_options(){
        let lap=Lapack::load().unwrap();let model=Model{dimension:2,states:vec![],h_real:vec![0.2,0.3,0.3,-0.4],h_imag:vec![0.;4],collapse:vec![],
            initial_index:0,initial_real:Some(vec![0.5,0.,0.,0.5]),initial_imag:Some(vec![0.;4]),time_dependent:true,frequency_coordinates:None,
            drives:vec![Drive{omega:2.3,cos:Operator{real:vec![0.,0.04,0.04,0.],imag:vec![0.;4]},sin:Operator{real:vec![0.;4],imag:vec![0.,0.02,-0.02,0.]}}]};
        let g=build_generator(&model).unwrap();let ext=extend(&model,&g,1).unwrap();assert_eq!(max_abs(&sparse_product(&ext.matrix,&ext.initial)),0.);
        let result=run(&lap,&model,&options());
        for sample in 0..options().samples{let actual=density(&result,sample);let expected=[0.5,0.,0.,0.5];
            assert!(actual.iter().zip(expected).all(|(a,b)|(a.re-b).abs()<1e-14&&a.im.abs()<1e-14));}
        for invalid in [Options{duration_us:f64::NAN,..options()},Options{samples:1,..options()},Options{include_asymptotic:true,..options()},Options{steady_only:true,..options()}]{
            assert!(independent_finite_result(&lap,&model,&g,&invalid).is_err());
        }
        assert!(!independent_finite_retryable("Invalid supplied structured Floquet metadata"));
        assert!(!independent_finite_retryable("Hamiltonian is not Hermitian"));
    }
}
fn physical_operator_indices(
    model: &Model,
    g: &Generator,
    extended: &Extended,
) -> Result<Vec<usize>, String> {
    let support = hilbert_support(model)?;
    let mut included = vec![false; g.n];
    for i in support {
        included[i] = true;
    }
    let physical: Vec<_> = g
        .coords
        .iter()
        .enumerate()
        .filter_map(|(i, &coordinate)| match coordinate {
            Coordinate::Population(a) => included[a].then_some(i),
            Coordinate::Real(a, b) | Coordinate::Imag(a, b) => {
                (included[a] && included[b]).then_some(i)
            }
        })
        .collect();
    // This is the full Hermitian operator algebra of the invariant Hilbert
    // support in EVERY retained harmonic. Initial-coordinate reachability is
    // valid for propagating one rho, but can hide conserved observables and
    // must never be used to certify a unique periodic stationary state.
    let d = extended.physical_dimension;
    let mut indices = Vec::new();
    for harmonic in &extended.harmonics {
        for part in 0..if harmonic.imaginary { 2 } else { 1 } {
            for &i in &physical {
                indices.push(harmonic.offset + part * d + i);
            }
        }
    }
    let mut allowed = vec![false; extended.matrix.len()];
    for &i in &indices {
        allowed[i] = true;
    }
    for &i in &indices {
        if extended.matrix[i]
            .iter()
            .any(|&(row, value)| value != 0. && !allowed[row])
        {
            return Err(
                "Physical Hilbert support failed Fourier generator-invariance verification".into(),
            );
        }
    }
    Ok(indices)
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
    let inds = if full_drive_mode() {
        // This is an exact invariant coordinate space seeded by the specified
        // density. Only algebraic zero paths are absent. We compute its phase
        // average, without asserting uniqueness outside that initial orbit.
        reachable(&extended.matrix, &extended.initial)
    } else {
        physical_operator_indices(model, g, &extended)?
    };
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
            indices:inds.clone(),
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
    let static_factory:Option<StationaryFactory<'_>>=if full_drive_mode()&&std::env::var("ATOM_STATIC_SYLVESTER_MEAN").is_ok_and(|value|value=="1"){
        let alpha=std::env::var("ATOM_STATIC_SYLVESTER_ALPHA").ok().and_then(|v|v.parse::<f64>().ok()).filter(|v|v.is_finite()&&*v>0.).unwrap_or(1e-3);
        let blocks:Vec<_>=extended.harmonics.iter().map(|h|(h.offset,h.index.iter().zip(&extended.frequency_basis.frequencies).map(|(&k,w)|k as f64*w).sum(),h.imaginary)).collect();
        let mapping=inds.clone();let full_dimension=extended.matrix.len();
        Some(Box::new(move|lap,a|{
            let inverse=std::rc::Rc::new(crate::stationary_sylvester::Prepared::new(lap,model,alpha,&blocks,full_dimension,&mapping,trace)?);
            let forward=std::rc::Rc::clone(&inverse);
            crate::block_solve::Prepared::with_custom(a,Box::new(move|lap,rhs|forward.apply(lap,rhs,false)))
                .map(|p|p.with_custom_transpose(Box::new(move|lap,rhs|inverse.apply(lap,rhs,true))))
        }))
    }else{None};
    let quick = if full_drive_mode() {
        block_stationary_solve(
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
            if std::env::var("ATOM_STRUCTURED_MEAN_PRODUCT").is_ok_and(|value|value=="1"){
                match structured_stationary_products(model,&extended,&inds,trace){
                    Ok(products)=>Some(products),
                    Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Structured stationary products unavailable: {error}");}None},
                }
            }else{None},
            if std::env::var("ATOM_REUSE_MEAN_ILUTP").is_ok_and(|value|value=="1"){
                Some(refinement_layout(&extended,&inds,&vec![1.;extended.matrix.len()])?)
            }else{None},
            static_factory,
        )
        .map_err(|error| {if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Stationary K={order} block fallback: {error}");}error}).ok()
    } else {
        None
    };
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
        indices:inds,
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
    let indices=reachable(&extended.matrix,&extended.initial);let dimension=indices.len();
    let size = extended.matrix.len();
    Stationary {
        extended, coefficients: vec![0.; size], residual: 0., boundary_residual: 0.,
        trace_error: 0., positivity_error: 0., dimension, condition: f64::INFINITY,
        valid: false, proof_cached: false, symbolic_cached: false, pending_condition: None,prepared:None,indices,
    }
}

fn stationary_linear_pass(current:&Stationary)->bool{
    current.trace_error<1e-8&&current.residual<1e-5
}
fn stationary_numerics_pass(current:&Stationary)->bool{
    stationary_linear_pass(current)&&current.positivity_error<1e-8
}
fn stationary_harmonics_pass(current:&Stationary,delta:f64,mean_delta:f64)->bool{
    delta.max(mean_delta)<(if full_drive_mode(){1e-6}else{1e-8})&&current.boundary_residual<1e-7
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
    let indices=if full_drive_mode(){reachable(&current.extended.matrix,&current.extended.initial)}else{physical_operator_indices(model,g,&current.extended)?};
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
        let prepared=if std::env::var("ATOM_STRUCTURED_MEAN_PRODUCT").is_ok_and(|value|value=="1"){
            let(forward,transpose)=structured_stationary_products(model,&current.extended,&indices,trace)?;
            prepared.with_raw_product(forward).with_raw_transpose(transpose)
        }else{prepared};
        std::rc::Rc::new(prepared)
    };
    let mut rhs=vec![0.;n];rhs[trace]=1.;let mut values:Vec<_>=indices.iter().map(|&i|current.coefficients[i]).collect();
    current.pending_condition=Some(std::rc::Rc::clone(&prepared));current.prepared=Some(std::rc::Rc::clone(&prepared));current.condition=f64::NAN;current.proof_cached=false;
    for round in 0..2{
        if round==1&&prepared.is_harmonic_extension()&&!prepared.has_fresh_native(){
            if let Err(error)=prepared.refresh_native_for_refinement(){
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Stationary K={order} fresh refinement factor unavailable; retry original equation with existing factor: {error}");}
            }
        }
        let(before,after)=stationary_raw_correction(lap,&prepared,&a,&rhs,&mut values)
            .map_err(|error|format!("Stationary K={order} same-order residual correction {} failed: {error}",round+1))?;
        for(&global,&value)in indices.iter().zip(&values){current.coefficients[global]=value;}
        refresh_stationary_checks(lap,model,g,current)?;
        records.push(json!({"round":round+1,"normalizedRawBefore":before,"normalizedRawAfter":after,"physicalRawResidual":current.residual,
            "traceError":current.trace_error,"positivityError":current.positivity_error,"boundaryForcingMax":current.boundary_residual,"freshNative":prepared.has_fresh_native()}));
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Stationary K={order} residual correction {}: normalized {before:e}->{after:e}, raw={:e}, trace={:e}, PSD={:e}, boundary={:e}",round+1,current.residual,current.trace_error,current.positivity_error,current.boundary_residual);}
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

pub(super) fn solve(lap: &Lapack, model: &Model, options: &Options) -> Result<Value, String> {
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
        let reason="A frequency relation holds only after floating-point rounding; it cannot establish an infinite-time mean. Finite propagation retains the original independent frequencies.";
        if options.steady_only {
            return Err(reason.into());
        }
        let mut result = solve_inner(
            lap,
            model,
            &Options {
                include_asymptotic: false,
                ..*options
            },
            validation,
        )?;
        if options.include_asymptotic {
            result["asymptoticStatus"] = json!("not-converged");
            result["diagnostics"]["asymptoticStatus"] = json!("not-converged");
            result["diagnostics"]["asymptoticReason"] = json!(reason);
        }
        result["diagnostics"]["unmergedRoundedFrequencyRelations"] = json!(basis.rounded_relations);
        result["diagnostics"]["rejectedFrequencyReconstructionErrors"] =
            json!(basis.reconstruction_errors);
        result["diagnostics"]["frequencyCoordinateSource"]=json!(basis.source());
        return Ok(result);
    }
    if options.steady_only && options.include_asymptotic {
        let cached = STEADY_RESULTS.with(|cache| {
            let mut cache = cache.borrow_mut();
            let index = cache.iter().position(|(key, value)| {
                same_model(key, model)
                    && value["diagnostics"]["fullDriveMode"]
                        .as_bool()
                        .unwrap_or(false)
                        == full_drive_mode()
            });
            index.map(|i| cache.remove(i).unwrap())
        });
        if let Some(entry) = cached {
            let mut value = entry.1.clone();
            value["diagnostics"]["stationaryCached"] = json!(true);
            value["diagnostics"]["elapsedMs"] = json!(0.);
            STEADY_RESULTS.with(|cache| cache.borrow_mut().push_back(entry));
            return Ok(value);
        }
    }
    let mut result = solve_inner(lap, model, options, validation)?;
    result["diagnostics"]["frequencyCoordinateSource"]=json!(basis.source());
    if let Some(mapping)=&basis.source_to_basis{result["diagnostics"]["carrierSourceToBasis"]=json!(mapping);}
    result["diagnostics"]["fullDriveMode"] = json!(full_drive_mode());
    if options.steady_only && options.include_asymptotic {
        STEADY_RESULTS.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.push_back((model.clone(), result.clone()));
            while cache.len() > 12 {
                cache.pop_front();
            }
        });
    }
    Ok(result)
}
fn solve_inner(lap: &Lapack, model: &Model, options: &Options, g: Generator) -> Result<Value, String> {
    let started = Instant::now();
    if full_drive_mode()&&!options.steady_only&&!options.include_asymptotic
        &&std::env::var("ATOM_INDEPENDENT_FINITE").is_ok_and(|value|value=="1"){
        return independent_finite_result(lap,model,&g,options);
    }
    if full_drive_mode()&&std::env::var_os("ATOM_SCHUR_BLOCKS").is_some()&&model.drives.len()==1{
        let shared=crate::schur_shift::Shared::new(lap,&g.matrix,g.n)?;
        SHARED_SCHUR.with(|s|*s.borrow_mut()=Some(std::rc::Rc::new(shared)));
    }
    let mut previous: Option<Stationary> = None;
    let mut previous_trajectory: Option<Trajectory> = None;
    let mut previous_mean: Option<Vec<f64>> = None;
    let mut trajectory_failures=Vec::new();
    let mut mean_published=false;
    let mut trajectory_restarts=0;
    let first_order=if full_drive_mode(){std::env::var("ATOM_MEAN_START_ORDER").ok().and_then(|value|value.parse::<i32>().ok()).filter(|value|*value>=1&&*value<=11).unwrap_or(1)}else{1};
    for order in first_order..=12 {
        let mut current = match stationary_candidate(lap, model, &g, order, !full_drive_mode()) {
            Ok(value) => value,
            Err(error) => {
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                    eprintln!("Periodic stationary solve needs initial projection: {error}");
                }
                let extended = extend(model, &g, order)?;
                unresolved_stationary(extended)
            }
        };
        let (mut mean, mut delta, mut mean_delta, mut mean_converged) = stationary_progress(
            lap, model, &g, options, &current, previous.as_ref(), previous_mean.as_ref(),
        )?;
        if full_drive_mode()&&stationary_requires_correction(&current,delta,mean_delta){
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic K={order}: harmonics converged but candidate failed (density={delta:e}, mean={mean_delta:e}, boundary={:e}, raw={:e}, trace={:e}, PSD={:e}); correct the SAME equation",current.boundary_residual,current.residual,current.trace_error,current.positivity_error);}
            refine_stationary_candidate(lap,model,&g,&mut current,order)?;
            (mean,delta,mean_delta,mean_converged)=stationary_progress(lap,model,&g,options,&current,previous.as_ref(),previous_mean.as_ref())?;
        }
        if mean_converged && current.pending_condition.is_some() {
            // Only the orders capable of contributing a returned mean or
            // trajectory pay for the full inverse-norm estimate. If that
            // check fails, retain the original complete-LU/nonunique fallback.
            if let Err(error) = certify_stationary_candidate(lap, &mut current, order) {
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                    eprintln!("Deferred stationary K={order} condition fallback: {error}");
                }
                current = match stationary(lap, model, &g, order) {
                    Ok(value) => value,
                    Err(error) => {
                        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some() {
                            eprintln!("Periodic stationary solve needs initial projection: {error}");
                        }
                        unresolved_stationary(extend(model, &g, order)?)
                    }
                };
                (mean, delta, mean_delta, mean_converged) = stationary_progress(
                    lap, model, &g, options, &current, previous.as_ref(), previous_mean.as_ref(),
                )?;
                if full_drive_mode()&&stationary_requires_correction(&current,delta,mean_delta){
                    refine_stationary_candidate(lap,model,&g,&mut current,order)?;
                    (mean,delta,mean_delta,mean_converged)=stationary_progress(lap,model,&g,options,&current,previous.as_ref(),previous_mean.as_ref())?;
                    if mean_converged&&current.pending_condition.is_some(){certify_stationary_candidate(lap,&mut current,order)?;}
                }
            }
        }
        if full_drive_mode()&&current.valid&&stationary_harmonics_pass(&current,delta,mean_delta)&&stationary_linear_pass(&current)&&!stationary_numerics_pass(&current){
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic K={order}: raw/trace gates passed, PSD={:e} still requires further Fourier refinement; no additional linear correction or factor refresh",current.positivity_error);}
        }
        if full_drive_mode()&&!options.steady_only&&current.valid&&!mean_converged{
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic K={order}: mean not converged (density={delta:e}, mean={mean_delta:e}, boundary={:e}, raw={:e}, trace={:e}, PSD={:e}); defer trajectory",current.boundary_residual,current.residual,current.trace_error,current.positivity_error);}
            current.pending_condition = None;
            previous=Some(current);previous_mean=mean;continue;
        }
        if full_drive_mode()&&mean_converged&&current.valid&&!options.steady_only&&options.include_asymptotic&&!mean_published{
            if !current.condition.is_finite() { return Err("An uncertified stationary candidate cannot be published".into()); }
            let x=mean.as_ref().ok_or("Converged mean is missing")?;
            let rho=expand_density(&g.coords,x,g.n);
            publish_mean(json!({"timesUs":[0.],"states":model.states,"populations":[g.initial[..g.n].to_vec()],
                "scatteringRatePerUs":[scatter_rate(&g.scattering,&expand_density(&g.coords,&g.initial,g.n),g.n)],
                "asymptoticPopulations":x[..g.n].to_vec(),"asymptoticDensityReal":rho.iter().map(|z|z.re).collect::<Vec<_>>(),"asymptoticDensityImag":rho.iter().map(|z|z.im).collect::<Vec<_>>(),
                "asymptoticFourierDensity":fourier_density(&g,&current.extended,&current.coefficients),"asymptoticStatus":"initial-dependent-mean",
                "diagnostics":{"engine":"rust-floquet","method":"Rust / converged full-density Fourier mean","asymptoticStatus":"initial-dependent-mean","densityLimitExists":false,"stationaryOnly":true,"initialDependent":true,"harmonicOrder":order,"harmonicConvergenceError":delta.max(mean_delta),"boundaryForcingMax":current.boundary_residual,"elapsedMs":started.elapsed().as_secs_f64()*1000.,"traceError":current.trace_error,"positivityError":current.positivity_error,"residual":current.residual,"stationaryProofCached":false,"asymptoticReason":"Converged long-time phase average; finite-time evolution is still running."}}));
            mean_published=true;
        }
        let mut affine_order=None;let mut affine_delta=None;let mut affine_dimension=None;
        let transient=if options.steady_only{None}else if full_drive_mode()&&mean_converged&&current.valid
            &&affine_center_requested(model,&current.extended,options){
            let refined=affine_refine_trajectory(lap,model,&g,&current,order,options)?;
            affine_order=Some(refined.order);affine_delta=Some(refined.delta);affine_dimension=Some(refined.dimension);
            trajectory_failures.extend(refined.failures);trajectory_restarts+=refined.restarts;
            Some(refined.trajectory)
        }else{
        let had_seed=RECYCLED_SPACE.with(|space|space.borrow().is_some());
        let mut attempt=trajectory(lap,model,&g,&current,options);
        if full_drive_mode()&&had_seed&&attempt.as_ref().err().is_some_and(|e|e.contains("Unstable projected exponential")){
            RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);trajectory_restarts+=1;
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic K={order}: discard unstable recycled basis and restart this order");}
            attempt=trajectory(lap,model,&g,&current,options);
        }
        match attempt{
            Ok(value)=>Some(value),
            Err(error) if full_drive_mode()&&(error.contains("Unstable projected exponential")||error.contains("Krylov space did not converge"))=>{
                trajectory_failures.push(json!({"order":order,"reason":error}));
                if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic K={order}: trajectory rejected: {error}");}
                if trajectory_failures.len()>=3{return Err(format!("Time evolution remained unstable across three Fourier refinements: {error}"));}
                RECYCLED_SPACE.with(|space|*space.borrow_mut()=None);previous_trajectory=None;previous=Some(current);previous_mean=mean;continue;
            },
            Err(error)=>return Err(error),
        }};
        let trajectory_delta = if let Some(delta)=affine_delta{delta}else{match (&transient, &previous_trajectory) {
            (Some(a), Some(b)) => a
                .states
                .iter()
                .zip(&b.states)
                .flat_map(|(a, b)| {
                    a[..if full_drive_mode() {
                        model.dimension
                    } else {
                        a.len()
                    }]
                        .iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs())
                })
                .fold(0., f64::max),
            (None, _) => 0.,
            _ => f64::INFINITY,
        }};
        if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic K={order}: density={delta:e}, mean={mean_delta:e}, trajectory={trajectory_delta:e}, boundary={:e}",current.boundary_residual);}
        // Full-model population accuracy targets 1e-3. Keep harmonic density
        // refinement at 1e-6 and trajectory refinement at 5e-5 instead of
        // spending another entire harmonic solve to resolve invisible digits.
        // Residual, boundary forcing, trace and positivity checks are unchanged.
        if delta.max(mean_delta) < (if full_drive_mode() { 1e-6 } else { 1e-8 })
            && trajectory_delta < (if full_drive_mode() { 5e-5 } else { 1e-8 })
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
            let mut trace_error = current.trace_error;
            let mut positivity_error = current.positivity_error;
            let (times, populations, scattering) = if let Some(tr) = &transient {
                let mut pops = vec![];
                let mut rates = vec![];
                for x in &tr.states {
                    let rho = expand_density(&g.coords, x, g.n);
                    trace_error = trace_error.max((x[..g.n].iter().sum::<f64>() - 1.).abs());
                    positivity_error = positivity_error.max(positivity(lap, &rho, g.n)?);
                    pops.push(x[..g.n].to_vec());
                    rates.push(scatter_rate(&g.scattering, &rho, g.n));
                }
                (tr.times.clone(), pops, rates)
            } else {
                (
                    vec![0.],
                    vec![g.initial[..g.n].to_vec()],
                    vec![scatter_rate(
                        &g.scattering,
                        &expand_density(&g.coords, &g.initial, g.n),
                        g.n,
                    )],
                )
            };
            if trace_error > 1e-7 || positivity_error > if full_drive_mode() { 1e-4 } else { 1e-7 }
            {
                return Err(format!("Fourier trajectory density verification failed: trace {trace_error:e}, positivity {positivity_error:e}"));
            }
            let asym = if options.include_asymptotic {
                mean.as_ref().map(|x| x[..g.n].to_vec())
            } else {
                None
            };
            let status = if !options.include_asymptotic {
                "not-requested"
            } else if current.valid {
                if full_drive_mode() {
                    "initial-dependent-mean"
                } else {
                    "periodic-mean"
                }
            } else if mean.is_some() {
                "initial-dependent-mean"
            } else {
                "unresolved-initial-dependent"
            };
            let density = if options.return_density {
                transient.as_ref().map(|tr| {
                    tr.states
                        .iter()
                        .map(|x| expand_density(&g.coords, x, g.n))
                        .collect::<Vec<_>>()
                })
            } else {
                None
            };
            return Ok(
                json!({"timesUs":times,"states":states,"populations":populations,
                    "densityReal":density.as_ref().map(|rows|rows.iter().map(|row|row.iter().map(|z|z.re).collect::<Vec<_>>()).collect::<Vec<_>>()),
                    "densityImag":density.as_ref().map(|rows|rows.iter().map(|row|row.iter().map(|z|z.im).collect::<Vec<_>>()).collect::<Vec<_>>()),
                    "scatteringRatePerUs":scattering,
                    "asymptoticFourierDensity":if full_drive_mode()&&options.include_asymptotic&&current.valid{Some(fourier_density(&g,&current.extended,&current.coefficients))}else{None},"asymptoticDensityReal":rho.as_ref().map(|rho|rho.iter().map(|z|z.re).collect::<Vec<_>>()),"asymptoticDensityImag":rho.as_ref().map(|rho|rho.iter().map(|z|z.im).collect::<Vec<_>>()),"asymptoticPopulations":asym,"asymptoticScatteringRatePerUs":rho.as_ref().map(|rho|scatter_rate(&g.scattering,rho,g.n)),
                    "asymptoticStatus":status,"diagnostics":{"method":"Rust / faer full-density Fourier-Galerkin Lindblad solve",
                        "engine":"rust-floquet","liouvilleDimension":g.n*g.n,"reachableDimension":affine_dimension.unwrap_or_else(||reachable(&current.extended.matrix,&current.extended.initial).len()),"harmonicOrder":affine_order.unwrap_or(order),"meanHarmonicOrder":order,"affineReferenceCenter":affine_order.is_some(),
                        "stationaryOperatorDimension":if current.valid{Some(current.dimension)}else{None},
                        "stationaryProofCached":current.proof_cached,"stationarySymbolicCached":current.symbolic_cached,
                        "uniquenessScope":if full_drive_mode(){"phase-average solution on the exact invariant coordinate space of the specified initial density; no global uniqueness asserted"}else if current.valid{"full Hermitian operator algebra on the invariant physical Hilbert support in every retained Fourier harmonic"}else{"initial-state spectral temporal-mean projection; no uniqueness asserted"},
                        "trajectoryBasisRestarts":trajectory_restarts,"trajectoryRejectedOrders":trajectory_failures,"harmonicConvergenceError":delta.max(trajectory_delta).max(mean_delta),"independentBeats":current.extended.frequency_basis.frequencies.len(),"stationaryOnly":options.steady_only,"initialDependent":full_drive_mode()||!current.valid,
                        "boundaryForcingMax":current.boundary_residual,"frequencyBasis":current.extended.frequency_basis.frequencies,
                        "stationarySystemConditionEstimate":if current.valid{Some(current.condition)}else{None},
                        "frequencyReconstructionErrors":current.extended.frequency_basis.reconstruction_errors,"roundedFrequencyRelations":current.extended.frequency_basis.rounded_relations,
                        "krylovDimension":transient.as_ref().map(|t|t.basis_order),"krylovConvergenceError":transient.as_ref().map(|t|t.convergence_error),
                        "elapsedMs":started.elapsed().as_secs_f64()*1000.,"traceError":trace_error,"positivityError":positivity_error,
                        "hermiticityError":0.,"residual":current.residual,"asymptoticStatus":status,"densityLimitExists":false,
                        "asymptoticReason":"Converged harmonic solution of the full driven Lindblad equation; populations are the long-time phase average, not a time-independent density."}}
                ),
            );
        }
        previous = Some(current);
        previous_trajectory = transient;
        previous_mean = mean;
    }
    Err("Fourier harmonic refinement did not converge within order 12; no stationary mean is asserted".into())
}
