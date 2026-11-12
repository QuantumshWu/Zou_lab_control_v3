//! Units, their existence gates, their measured errors (SPEC §1.5, §5.2).
//!
//! # One rule
//! No light is far or near. Every drive -- one component's coupling of one ground state to one excited state -- and every
//! second-order coupling of two states at one beat is a unit of its own, judged alone by one statement at its own distance:
//! a fold of a coupling of size `size` across a distance `distance` exists where m = size/distance < ½, S = 2m² < ½
//! (Stewart 1973, internal.md 1B.1; `EXISTS`), and a moving atom's fold is still the instantaneous one's to first order
//! where a = rate/distance² < ½, rate = |dΔE/dt| -- the gate at the same relative error as the static one (internal.md
//! 1B.1), not an existence theorem. `gate` is that statement, the only one; a kind of unit supplies its own size, its least
//! distance over the reach and its rate, all provable bounds over the reach (reach.rs): a coupling's coherent peak, every
//! beam at the largest amplitude its cell allows and all in phase, over the distance's least, the energies' spread and
//! every Doppler shift of the reach's speeds taken against it. Neither condition implies the other; a fold needs both.
//!
//! The sweep is a relative error, of the fold's own effect: the instantaneous fold of a linear sweep is the exact
//! response times ∫₀^∞ e^{−x + iax²/2} dx, of modulus at most one at any a (internal.md 1B.1), so a moving fold is off
//! by at most twice its own amplitude m, (2m)² = 2S of its states' population, however fast it sweeps. So the sweep is
//! read against the larger of that effect and the least share any output resolves, unseen(2¹⁴) (SPEC §5.4: the share
//! the ledger cannot feel): a · min(1, 2S/unseen) (`Gate::resolved`) -- the sweep itself where the fold's effect is
//! resolved, scaled down with it below. (A coupling of 1e-9 of its line's, which the H0 states' mixing gives the repump
//! between Zeeman sublevels two apart, swept at the tweezer's light-shift gradient read a = 0.5000 over the reach of the
//! user's lin⊥lin probe, and its proof was 95 % of 133 s. Cut off at 2S = unseen instead, the quantity jumped there:
//! a coupling's bound over a cell, its S a little above the cutoff, kept its sweep of 0.8 to 8 while the cell's centre,
//! a little below, read 0 -- a bound that no longer tends to its point value, and the proof never ended.)
//!
//! Keeping a unit is always exact -- the more exact level -- so a proof over the reach only has to settle the fold:
//! below ½ everywhere it holds; a unit whose proof meets a centre in the reach attaining ¼ first is kept (`KEEP`,
//! reach.rs # Provable extrema). Within [¼, ½) both are right: a fold attaining S there books its mixing S/2, an eighth
//! of its states' population or more (# The errors), 2.5 times the population target, which its states cannot meet
//! where they are held -- keeping it is what the error asks anyway (SPEC §5.2's first rule). The cells then shrink to
//! the band's width at most, not to a value's margin from ½, whose inverse cube their number grows as: lin⊥lin's
//! couplings of F=1 m=0 with m=±1 at 0.497 and 0.499 took 1.1 million cells.
//!
//! # Levels
//! A drive is taken at one of three levels (model.rs `Level`), each the more exact one of the one before, and what decides
//! is the accuracy its own denominator needs -- never its frequency, its line or its beam:
//! - folded with the bare levels (`Unit::Bare`): its response is V/(Δ₀ − iΓ/2), Δ₀ = E_e − E_g − ν of H0, a constant
//!   (far.rs). At the atom the denominator is Δ₀ + δΔ(r, v) − iΓ/2 -- δΔ the shift of its two levels by the other light
//!   and the Doppler shift and recoil of its photon --, so the bare response is the first term of a series in
//!   ε = δΔ/(Δ₀ − iΓ/2), and the bare levels are a perturbation theory about H0: what it folds across the bare distance
//!   |Δ₀ − iΓ/2| is everything between the drive's two states that H0 does not hold, the coupling and the movement of the
//!   two levels together. Its own gate is therefore the one gate of the size |V| + |δΔ|, whichever way the levels move,
//!   across the bare distance: below it the fold exists (|V| against what is left of the distance, |Δ₀ − iΓ/2| − |δΔ|) and
//!   the series converges with |ε| < ½; failing, the unit is kept and the drive is folded with the levels at the atom.
//!   Inside the gate its error is the series' first order, measured at the atoms (# The errors).
//! - folded with the levels at the atom (`Unit::Drive`, eliminate.rs): the levels at the atom hold the atom at rest --
//!   its energies, the dressings, the self-energy at its denominator D --, and a moving atom's photon of beam l is
//!   Doppler-shifted by w_l = (ħ/m)(k_l·κ + k_l²/2), which the response takes to first order, T − w T2 (eliminate.rs,
//!   # First-order Doppler): the first two terms of the series in ε = w/D of (D + w)⁻¹. Two expansions, each with its
//!   own small parameter: the fold of |V| across the moving atom's actual denominator (its distance over a region, the
//!   Doppler shift an interval of its detuning: its share, # The tail rule), and the series in ε about the atom at rest.
//!   Each holds where the one gate says it does -- the tail of the shares below ½, and the series' own gate, the one gate
//!   of |w| across D (`Across::distance_at_rest`; |w| ≤ k v + (ħ/m) k²/2), below ½: |ε| < ½, the margin a bare drive's
//!   series has --, and the drive's gate is the larger of the two. The share is no gate of the series: for |V| < Γ/4 it
//!   is below ½ at any speed, the actual distance never below Γ/2, while |w/D| > 1 diverges -- the released runaway, a
//!   resonant probe's cycling drive folded at the atom with |w/D| ≈ 85 at the atoms it pushed out, 10⁵ jumps/µs. (The
//!   one gate of |V| + |w| across D, the bare level's construction, says both but asks more than either: (|V| + |w|)/D
//!   is above |V| across the actual distance by about (|w|/D)(1 − |V|/D) and above |ε| by |V|/D -- on S2's repump at the
//!   atom over its 30 µK reach 0.4998 where the share reads 0.47, and the reach's proof, a threshold at its quantity's
//!   largest value, ran out of memory.) Its gate failing, the unit is kept and the drive is explicit.
//! - explicit: in the generator as it is.
//!
//! A drive folded onto an excited state that another drive makes explicit is, at either level, one leg of a Raman pair
//! with each kept drive V_a of that state (far.rs, # A fold onto an explicit state): beside its own gate it has the
//! pair's (`Model::pair_gate`), the one gate of the two-photon coupling |U V_a| over the nearer of the two one-photon
//! distances -- the amplitude either of the pair's states holds of the other is that over the two-photon distance --
//! across the two-photon distance |E_g + ν − E_a − ν_a| over the region, less the two-photon Doppler shift and what the
//! folded and the kept drives move the two ground levels. The pair's coupling U V_a* turns in space as e^{i(q − k')·r},
//! q the folded drive's photon (k_l, a counter-rotating component's −k_l) and k' the kept one's, so a moving atom's
//! two-photon detuning moves by (q − k')·v: at most |q − k'| v over the two components' beams -- nothing for two photons
//! of one beam, 2kv for counter-propagating ones (`Model::two_photon_wave`). Its centre moves over the region as the
//! difference E_g − E_a does, a shift the two levels share moving nothing. (Taken as |q| + |k'| for any two beams and
//! each level's own movement, the bound of two legs of the one repump beam between F=1 m=1 and m=0, 2.1 MHz apart,
//! reached zero at an atom of 0.53 µm/µs: the pair's gate read ∞ for a two-photon coupling |U V|/d of 1e-18 rad/µs.)
//! The pair is the second-order coupling of two slow states -- g and the state a of e's cluster --, as a coupling of two
//! states is (# A coupling of two states), not an amplitude of an excited state of g's: its own statement, judged alone
//! (# The tail rule); failing, the drive is kept as any other.
//!
//! A drive has exactly one level, the highest its kept units give it, and one more where its two states are one cluster's
//! at a residual beat of exactly zero (model.rs `Model::settled`); a carrier is a component with a drive not bare, never listed.
//!
//! # The tail rule
//! The folds of one ground state's drives share its population: the fold of a set of them exists where their shares'
//! sum is below ½, each share across its drive's actual distance -- whatever level it is computed at. A drive's gate is
//! therefore the sum of its ground state's folded shares that are not larger than its own -- so that the strongest fails
//! first and is kept first, drives of one share fail together, and a distant line is never kept for a near one's sake --,
//! and a bare drive's takes its own gate's value for its share: that its bare fold exists is its own matter and no other
//! drive's -- as is a drive at the atom's Doppler series, its gate beside the tail (# Levels). The order is one fixed
//! order per ground state, by each drive's share with every
//! beam at its peak and the bare levels (`Model::ranked`): a region's shares in a region's order would not be monotone in
//! the region, and the proof over the reach needs that; which of two is kept first is a choice of cost, each gate a
//! statement about a fixed set. Two shares within the rounding of their own arithmetic are one place in the order (drives
//! equal by a symmetry of the request differ in the last bits of their couplings), and its drives of one level take
//! their largest sweep. Over a cell, and on a point, every share of the sum is that cell's or that point's own: a
//! point's gate is the value there. (A weaker place's bound from elsewhere in its stead is no value of any point: with
//! the whole space's 0.49 for one of two equal drives, the other's gate read 0.52 "attained" at the trap's centre where
//! the two shares are 0.03 each, and a light 20 linewidths off its line was made explicit.)
//!
//! What the tail sums is what g's fold holds of the excited states its folded drives reach, the first-order amplitudes
//! R[e, g] = V_e/D_e: the column whose norm bounds the fold of g's set (Stewart, the block of one ground state against
//! its folded excited states). A Raman pair's mixing is none of them: it is the amplitude of another slow state in g's,
//! of the second order, |U V_a|/(d d_R), and as d_R → 0 it has no bound against the drive's own share |U|²/d² -- so it has
//! no place in an order by the drives' own shares. A drive's gate is therefore the larger of its place's tail and of its
//! place's drives' pairs (their largest, as their sweeps). (Added at its drive's place, a pair failed every stronger drive
//! of g with it, which keeping does not cure -- the pair stays until its own drive is kept --, down to the
//! counter-rotating ones, which can only be folded: the request of a lin⊥lin probe with its repump was refused for a
//! pair of the repump's 1.7e-9 rad/µs coupling F=1 m=1 → F′=2 m=−1 -- made by the H0 states' mixing -- with the
//! 1.1e-8 one from F=1 m=0, measured.)
//!
//! # What a folded drive does to its level
//! A drive of coupling v folded across the distance d moves its ground level by at most v²/d = m v, and by less than
//! v/2 wherever its fold exists: the shift is the fold's own second order, across the fold's own distance. So what the
//! drives folded at the atom do to a ground level over a region is bounded drive by drive, each across the distance its
//! own gate reads there (`Model::across`, `Model::near_shift`), and the slope of that shift with the response's norm
//! 1/d of the nearest of them (eliminate.rs `Model::shift_slope`). Never across Γ/2, the distance of a resonant drive,
//! whatever the drive: that bound is 2|Δ|/Γ times the shift of a light Δ off its line, grows with the intensity as the
//! shift does, and at 80 mW/cm² of light 20 linewidths off it exceeded the 137 MHz between a cooling and a repump
//! photon -- a coupling's distance read nothing, its gate ∞, and F=1 and F=2 were made one cluster.
//!
//! # The fold's order
//! The second-order fold leaves out one term of the next order, the fourth-order light shift: a state's potentials are
//! too deep by s/2 and the force on it too large by s, s = 2 × its excited share; its photons come too fast by s
//! (exactly, on one line: the force (1 − s + 3/2 s²) and the rate 1/(1 + s) of the second order's). T follows the
//! potentials' modulation, 0.8 to 0.9 s on a Sisyphus pattern -- the diffusion +0.6 s, the friction −0.2 s, both of the
//! one term, not two errors to add --: the force's s bounds it, and that is booked, s p_g on T and on the populations.
//! Where the fold has a fourth order (assemble.rs, # The fourth order; model.rs `Model::fold_frequency`) that part is
//! its own unit, the cluster's (`Unit::Second`): kept, the fold is the fourth order's, the force's error 3/2 s² p_g and
//! the levels' spread over the drive's distance times s (W3's one denominator a column), and the rates' s² p_g
//! with the photon a jump's landing in the bare state draws (model.rs `Model::landed`, step.rs `jump`). A drive's own
//! unit then carries what only keeping it explicit removes: that residue, and at either order its motion's part, the
//! sweep a and the Doppler term's next order. Elsewhere a drive's unit carries all of it.
//!
//! These are one line's. A ground state's drives of one carrier are one fold, not several: with D_e = E_e − E_b − δ_c −
//! iΓ_e/2 a drive's denominator and a_k = Σ_e |V_e|²/D_e^k, the state's slow eigenvalue is λ = −a₁(1 − a₂ + a₂² + a₁a₃ −
//! …) -- the second order N = −a₁, the fourth a₁a₂ (assemble.rs: N, Y = a₂ and X = Σ_e |R_e|² are these sums) --, and
//! every term the fourth order leaves, of λ, of the jumps' rate Σ_e Γ_e |R_e|² |1 + λ/D_e + …|²/(1 + X + …) and of the
//! landing's norm, is a product of two numbers of two kinds: the state's excited share, a₂ or X, at most S_b/2 = Σ_e
//! |R[e, b]|²; and its level's own second-order term over a drive's distance, a₁/D_e (the response's denominator moved
//! by the fold itself: a₁a₃ = Σ_e (V_e/D_e)² a₁/D_e), in modulus L_e/2 = |N_bb|/|D_e|. At one distance both are S_b/2 --
//! the excited states are one bright state, a two-level atom of the share S_b = Σ_e s_e, its rate the second order's
//! over 1 + S_b exactly -- and the one line's remainders stand with S_b for s: c S_b², not c Σ_e s_e². At different
//! distances a near line reads the far ones' light shift across its own distance, L_e ≤ Σ_f s_f |D_f/D_e|, which no
//! share bounds. So a drive's line errs at the fourth order by c max(S_b, L_e)², c the one line's (`ORDER_RATES`,
//! `ORDER_FORCE`), and its unit carries that on its part s_e/S_b of the state's excitation, c s_e max(S_b, L_e)²/S_b p_b
//! (eliminate.rs `Model::drive_error`); the potential's own remainder |a₂² + a₁a₃| ≤ Σ_e s_e (S_b + L_e)/4 is within
//! it. Against the exact waiting time of one ground state with one to three excited states (the test below): equal
//! shares at one detuning leave +0.87 to +1.07 (Σs)² on resonance and −0.98 (Σs)² at 6 Γ, two and three times Σs²;
//! shares of 0.04 at 3 Γ and at 30 Γ leave −7.2 (Σs)², 0.42 of what is booked. (An emulation of the same formulas on
//! 16 000 random sets of one to four lines from 0.01 Γ to 100 Γ on either side of the light, max(S_b, L_e) up to 0.2:
//! the rate's remainder is at most 1.09 Σ_e s_e max(S_b, L_e)²/S_b, the one resonant line's;
//! motion3d_status/rewrite/booking.status.md.)
//!
//! The second order's own error is made of the same two numbers, once each. A line's rate is |R_e|² |1 + λ/D_e + …|²/(1 +
//! X + …), and the second order takes |R_e|² alone: it errs by 2 Re(a₁/D_e), the level's own term over the line's
//! distance -- far from every line the rate is high by ⟨L_e⟩ exactly --, and by the shares X and a₂, which on one line
//! make up the same s. So the second order's s p_b is s_e max(S_b, L_e)/S_b p_b of a drive (`Folded::order`): Σ_e s_e p_b
//! where the drives are at one distance, more where a near line reads a far one's shift -- a far line that is folded at
//! the atom too: one folded with the bare levels moves the near one's denominator by its static form, which the levels
//! at the atom hold. Measured (the same test): at one detuning the rate is high by 1.000 Σs; shares of 0.04 at 3 Γ and at
//! 30 Γ, by 2.72 Σs (booked 3.21 Σs); 0.01 at 1 Γ with 0.05 at 40 Γ, by 3.10 Σs (5.83); 0.06, 0.006, 0.002 at 20, 64 and
//! 90 Γ, by 1.19 Σs (1.26). The force's error -- twice the potential's -- stays within Σs at any distances (0.94 Σs at
//! most there), within what T is booked. (The same emulation: the rate at most 1.000 Σ_e s_e max(S_b, L_e)/S_b, the one
//! line's, where Σs is exceeded 24 times.)
//!
//! L_e/S_b is taken as a ratio of distances, (|N_bb|/|D_e|)/Σ_f (|V_f R_f|/|D_f|) (eliminate.rs `Model::drive_parameters`):
//! 2|N_bb|/|D_e| over S_b where the responses are V/D, at most one at one distance, and one on a lone line whatever its
//! response holds beside. (With 2|N_bb|/|D_e| against S_b itself the response's first-order Doppler term, R = (V/D)(1 −
//! w/D) where the distance is |D + w|, made a lone line read L/S = 1/|1 − (w/D)²| -- 1.35 on R3_DOPPLER's fastest atoms:
//! the Doppler term's own next order, which its unit books, counted again.)
//!
//! # A coupling of two states (`Unit::Term`)
//! Everything the folds leave between two states a ← b at one exact beat β (∝ e^{iβt}) is one unit: the constant forms'
//! entry (far.rs `constant`: one frequency's static form at β = 0, with the two states' own coupling when every light is
//! off; two frequencies' cross form at their beat; on the ground states and on the excited ones) and the part of the
//! drives folded at the atom (eliminate.rs). Its size is the
//! bound of that entry, its distance |β + E_a − E_b| over the region less the Doppler shift of its beam pairs' wave
//! vector and the shifts of the two states by their drives folded at the atom (# What a folded drive does to its
//! level), its rate v|∇(E_a − E_b)| + q a_max. Kept, the entry is
//! the generator's and the two states are one cluster's, their frames the beat apart (`Model::edges`); folded, it is
//! dropped and its error booked. An entry at a residual beat of exactly zero in a cluster's frames is the generator's
//! without a unit.
//!
//! # The model's domain
//! A counter-rotating drive is only ever folded with the bare levels: one kept is refused, and so is a request whose
//! counter-rotating folds do not exist by the one gate, every beam at its peak (`Model::new`). A co-rotating leg against a
//! counter-rotating one is no term of the model (far.rs, # Two frequencies' constant form).
//!
//! # The errors (SPEC §1.5), measured at a trajectory state
//! What the units' errors at one state have in common is made once for all of them (`Measured`): the beams' amplitudes and
//! the point's region, the cluster's forms at the atom there -- one elimination, or its table's interpolant where the run
//! has tables -- with the carriers' responses and the bare drives', the constant forms' energies and their slopes, the
//! force on the state. Each unit's error is then a few numbers' arithmetic, nothing built per unit; only a carrier's
//! dressing takes an elimination of its own (the one with its coherent field). A unit's small parameter at the state's
//! point and momentum is mapped to the outputs by its leading order, the populations and T by the same weight -- the
//! populations of the states it couples, so that a unit whose states are empty books nothing: a drive's fold its own
//! share s p_g on the populations and (s + a) p_g on T (# The fold's order); a bare one also 2ε times its share of the state's rates, on
//! both, and ε times its share of the state's light shifts on T (the relative errors it makes of them, as a dressing's;
//! the shares of a state's drives, folded and kept, add up to one: beside an explicit line a distant one's rate is a
//! small part of what moves the populations); a coupling of two states m²(p_a + p_b); a carrier's dressing its relative rate error on the state's rates (eliminate.rs); a photon
//! coherence m²√(p_i p_j) (photon.rs).

use super::assemble::{NearValues, Plan};
use super::cluster::Edge;
use super::eliminate::term_unit;
use super::far::EXCITED;
use super::reach::{distance, pair_span, Cell, Region, TermForm};
use super::readout::Channels;
use super::tabulate::Tables;
use super::{Beat, Cluster, Component, Gate, Level, Model, Reach, Sample, Unit, UnitError, Wave};
use crate::field::Amplitude;
use crate::linalg::norm3;
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};
use crate::units::rounding;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Where a fold exists (module documentation): S = 2m² < ½ and a < ½.
pub(crate) const EXISTS: f64 = 0.5;

/// Where a gate's proof over the reach may end in keeping its unit (module documentation, # One rule): a value attained at
/// a centre in the reach at half the threshold, S = ¼ -- a fold there books its mixing S/2 = ⅛ of its states'
/// population, 2.5 times the population target 0.05.
pub(crate) const KEEP: f64 = 0.5 * EXISTS;

/// The fourth order's leading errors in one line's share s (module documentation, # The fold's order, where a ground
/// state's several drives take the line's place; derivation: fourth.status.md), each the next order's: on the force
/// 3/2 s², the next term of the one line's (1 − s + 3/2 s²); on the rates 9/8 s², the most that the slow state's
/// jumps with the landing's norm against the threshold (model.rs
/// `Model::landed`) leave of the exact 1/(1 + s) at any detuning. Far from the line −s² (the jumps and their prompt
/// photons, 1/(1 − p) a jump: −0.99 s² measured). On resonance the jumps with the exact mean interval (1 − p)/γ are
/// right to this order (+0.00 s²), and the landing's norm 1 − p = 1 + 3s/2 held against the threshold waits
/// ln(1 − p)/γ where that mean has −p/γ: the rate is high by p²/2 = (3/2)²/2 s² (two levels: +1.07 s² at s = 0.03,
/// between the two at the detunings between, the test below).
pub(crate) const ORDER_FORCE: f64 = 3.0 / 2.0;
pub(crate) const ORDER_RATES: f64 = {
    // −p/s on resonance, p = (s/2)(Δ² − 3Γ²/4)/(Δ² + Γ²/4) at Δ = 0; then p²/2.
    let landing = 3.0 / 2.0;
    landing * landing / 2.0
};

/// The leading error of a Raman pair's fold in its mixing m² (eliminate.rs `Model::pair_error`): the kept line reads
/// the folded leg's self-energy |U|²/(d_R + λ) at λ = 0, and its response's next term makes its rate m² (3ε² − Γ²/4)/(ε² +
/// Γ²/4) and its shift m² (ε² − 3Γ²/4)/(ε² + Γ²/4) of themselves -- at any detuning ε of the kept line at most 3 m².
pub(crate) const PAIR: f64 = 3.0;

impl Gate {
    /// Whether the fold exists over the reach: both bounds below ½.
    pub fn exists(&self) -> bool {
        self.saturation < EXISTS && self.sweep < EXISTS
    }

    /// The gate with its sweep as far as an output can see it (module documentation, # One rule): the sweep read against
    /// the larger of the fold's largest error from any motion, (2m)² = 2S of its states' population, and the least share
    /// any output resolves, unseen(2¹⁴) -- a · min(1, 2S/unseen), continuous and monotone in both, so that a bound over a
    /// cell tends to its point value.
    pub(crate) fn resolved(self) -> Gate {
        let seen = (2.0 * self.saturation / crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64)).min(1.0);
        Gate { saturation: self.saturation, sweep: if self.sweep == 0.0 { 0.0 } else { self.sweep * seen } }
    }

    /// The error of the unit this is the gate of, at most, at any state the reach holds -- where the unit's error is its
    /// gate's own parameter at the state (module documentation, # The errors), on the populations and on T alike. A
    /// coupling's (`Unit::Term`) is m²(p_a + p_b), m² = S/2 its mixing at the state's point and speed, which the gate
    /// bounds over the reach, and the two states' populations: together at most 1, and 2 p_a where a = b (a level's
    /// beating shift counts its one state on both sides). A photon coherence's is m²√(w_i w_j), m² = S/2 of its two
    /// classes' sizes there and w the largest population among a class's sources, at most 1. None for the others: a
    /// drive's error carries the first order of its denominator and its pair's mixing, which no gate of its own bounds;
    /// a dressing and a fold's order have no gate.
    pub fn error_bound(&self, unit: &Unit) -> Option<f64> {
        match unit {
            Unit::Term { a, b, .. } => Some(0.5 * self.saturation * if a == b { 2.0 } else { 1.0 }),
            Unit::Coherence { .. } => Some(0.5 * self.saturation),
            Unit::Bare { .. } | Unit::Drive { .. } | Unit::Second { .. } | Unit::Dressing { .. } => None,
        }
    }
}

/// The one gate (module documentation, # One rule): of a fold of a coupling of size `size` across a distance `distance`
/// that moves at `rate`, S = 2m², m = size/distance, and a = rate/distance² -- zero where there is nothing to fold or
/// nothing moves, infinite where nothing separates what does.
pub(crate) fn gate(size: f64, distance: f64, rate: f64) -> Gate {
    let over = |x: f64, d: f64| if x == 0.0 { 0.0 } else if distance > 0.0 { x / d } else { f64::INFINITY };
    let m = over(size, distance);
    Gate { saturation: 2.0 * m * m, sweep: over(rate, distance * distance) }
}

/// What the units' errors at one trajectory state share (module documentation, # The errors): the state; the beams'
/// amplitudes; the point's region at the state's speed; the cluster's plan under the running lights and its forms at the
/// atom at the point (the Hermitian ones to first order), with each carrier's folded response there; the bare drives'
/// responses there per component (readout.rs `Model::far_channels`); the state's excitation drive by drive -- Σ |R[e, g]|²
/// p_g over its folded drives, bare ones and those at the atom, with its explicit states' populations: its rates' measure;
/// and Σ |V[e, g] R[e, g]| p_g with its kept drives' |V[e, g]| |ψ_e| |ψ_g|: its light shifts' --; per carrier and ground
/// place of the cluster what that carrier's fold at the atom makes of the state (`Fold`);
/// the levels' energies at the atom to first order ([part][state], eliminate.rs `Model::far_energies`); the force on the
/// state (rad/µs/µm); its populations; and, made when asked, its cluster's photon classes' jump operators' sizes at the
/// point (photon.rs `Model::class_sizes`), per carrier the first-order Doppler term's size per ground place, and the
/// kept drives' couplings at the point with the ground levels' lifts by them (eliminate.rs `Model::kept_at`,
/// `Model::lifts_at`: what every drive folded onto an explicit state asks of the point, its pairs' legs).
pub(crate) struct Measured<'a> {
    pub(crate) sample: &'a Sample,
    pub(crate) amplitudes: Vec<Amplitude>,
    pub(crate) region: Region,
    pub(crate) plan: Arc<Plan>,
    pub(crate) near: NearValues,
    pub(crate) responses: Vec<Option<Mat>>,
    pub(crate) bare: Vec<Channels>,
    pub(crate) rates: f64,
    pub(crate) shifts: f64,
    pub(crate) folds: Vec<Vec<Fold>>,
    pub(crate) energies: Vec<Vec<f64>>,
    pub(crate) force: [f64; 3],
    pub(crate) populations: [f64; STATES],
    pub(crate) classes: OnceCell<Vec<[f64; 3]>>,
    pub(crate) doppler: Vec<OnceCell<Vec<f64>>>,
    pub(crate) kept: OnceCell<Vec<C64>>,
    pub(crate) lifts: OnceCell<[f64; GROUND]>,
}

/// What a carrier's fold at the atom makes of a ground state g of a sample's cluster (`Measured::folds`; the numbers the
/// fold's orders' errors are made of, module documentation, # The fold's order): its excited share S_g = 2 Σ_e |R[e, g]|²;
/// its level's own second-order term N_gg = Σ_e V[e, g]* R[e, g], its light shift and half its rate; each drive's size
/// in that, |V[e, g] R[e, g]| (by excited state less GROUND); and, made when a drive asks, the sum of those sizes each
/// across its drive's distance (eliminate.rs `Model::drive_parameters`).
#[derive(Clone)]
pub(crate) struct Fold {
    pub(crate) share: f64,
    pub(crate) level: C64,
    pub(crate) terms: [f64; EXCITED],
    pub(crate) across: OnceCell<f64>,
}

/// One drive of a ground state in the tail rule's order (module documentation, # The tail rule): its component and excited
/// state, and its place -- drives of one share within rounding share one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ranked {
    pub(crate) component: Component,
    pub(crate) excited: usize,
    pub(crate) group: usize,
}

/// One drive of a ground state over a region (`Model::across`): its coupling's size v, its detuning at the region's
/// centre Δ_c, how far from it its detuning lies at most over the region, and how far for an atom at rest -- all but
/// the Doppler shift of the region's speeds (`still`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Across {
    pub(crate) size: f64,
    pub(crate) centre: f64,
    pub(crate) moved: f64,
    pub(crate) still: f64,
}

impl Across {
    /// Its distance over the region, at least: |max(|Δ_c| − moved, 0) − iΓ_e/2|.
    pub(crate) fn distance(&self, gamma: f64) -> f64 {
        C64::new((self.centre.abs() - self.moved).max(0.0), 0.5 * gamma).abs()
    }
    /// Its distance over the region for an atom at rest, at least: |max(|Δ_c| − still, 0) − iΓ_e/2| -- the denominator
    /// a fold at the atom takes, whose response the Doppler shift enters to first order (module documentation, # Levels).
    pub(crate) fn distance_at_rest(&self, gamma: f64) -> f64 {
        C64::new((self.centre.abs() - self.still).max(0.0), 0.5 * gamma).abs()
    }
}

/// One kept drive over a region (`Model::kept_over`): its carrier, its excited state (less GROUND) and its ground state;
/// its coupling's coherent bound there and its one-photon distance at least (`Across::distance`, of its two levels' slack,
/// its Doppler shift and its excited state's Gershgorin radius).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Kept {
    pub(crate) carrier: usize,
    pub(crate) excited: usize,
    pub(crate) ground: usize,
    pub(crate) size: f64,
    pub(crate) distance: f64,
}

/// What a unit's gate reads of a region (`Model::judged`, `Model::gate_of`).
pub(crate) enum Read<'u> {
    Fold { ground: usize, place: usize, bare: bool },
    Term { form: Arc<TermForm>, a: usize, b: usize, beat: usize },
    Coherence { unit: &'u Unit, apart: f64 },
    Nothing,
}

/// A model's units' gates as functions of a region (`Model::judged`): what each unit's gate reads, in the units' order,
/// and the largest wave number of the light's beams twice, a photon coherence's two photons'. The proofs over a reach
/// (`Model::gates`) and the steps outside it (solve.rs `Domain`) read the same gates.
pub struct Judged<'u> {
    reads: Vec<Read<'u>>,
    wave: f64,
}

/// Two frequencies' constant form (far.rs `constant`): on their fields' bases and per pair of their beams, with the
/// rounding of an entry's own sums (reach.rs `TermForm::rounding_of`) -- the form's, whichever entry is read.
pub(crate) struct Cross {
    pub(crate) forms: Vec<Mat>,
    pub(crate) pairs: Vec<Mat>,
    pub(crate) rounding: f64,
}

impl Model {
    /// The spread of what moves the blocks' spectra, a bound over all space (rad/µs): the constant forms' coherent peaks
    /// and the carriers' couplings at theirs; made once (`Model::new`). The photon coherences' distance takes it (photon.rs).
    pub(crate) fn spread(&self) -> f64 {
        self.spread
    }
    pub(crate) fn spread_made(&self) -> f64 {
        let constant: f64 = self.tones.iter().map(|t| t.pairs.iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        let carriers: f64 = self.carriers.iter().map(|c| self.tones[c.frequency].couplings[0].iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        constant + carriers
    }

    /// A drive's denominator with the bare levels, its modulus |E_e − E_g − ν − iΓ_e/2| (rad/µs).
    pub(crate) fn bare_distance(&self, c: Component, excited: usize, ground: usize) -> f64 {
        let detuning = self.energies[excited] - self.energies[ground] - self.tones[c.frequency].nu[c.counter as usize];
        C64::new(detuning, 0.5 * self.decay[(excited - GROUND, excited - GROUND)].re).abs()
    }

    /// A drive's share anywhere: the one gate's S with every beam at its peak and in phase, |a_l| ≤ 1, across its bare
    /// distance.
    pub(crate) fn share_anywhere(&self, c: Component, excited: usize, ground: usize) -> f64 {
        let size: f64 = self.tones[c.frequency].couplings[c.counter as usize].iter().map(|x| x[(excited, ground)].abs()).sum();
        gate(size, self.bare_distance(c, excited, ground), 0.0).saturation
    }

    /// Per ground state its drives in the tail rule's order (module documentation, # The tail rule): by their shares
    /// anywhere, the largest first; two neighbours within the sum of their shares' roundings share a place. A share's
    /// relative rounding: its n beams' moduli and their sum (each entry the request's contraction with its field, three
    /// terms; γ_{n+5}), squared; the detuning's two differences, |E_e| + |E_g| + |ν| rounded once each against the
    /// denominator Δ₀² + Γ²/4 (2|Δ₀| u (|E_e| + |E_g| + |ν|) of it at most); the decay's sum over the ground states and
    /// the quotient.
    pub(crate) fn ranked_made(&self) -> Vec<Vec<Ranked>> {
        (0..GROUND)
            .map(|g| {
                let mut of: Vec<(Component, usize, f64, f64)> = self
                    .driven
                    .iter()
                    .filter(|d| d.2 == g)
                    .map(|&(c, e, _)| {
                        let tone = &self.tones[c.frequency];
                        let (nu, gamma) = (tone.nu[c.counter as usize], self.decay[(e - GROUND, e - GROUND)].re);
                        let detuning = self.energies[e] - self.energies[g] - nu;
                        let moved = 2.0 * rounding(1) * (self.energies[e].abs() + self.energies[g].abs() + nu.abs()) * detuning.abs() / (detuning * detuning + 0.25 * gamma * gamma);
                        let share = self.share_anywhere(c, e, g);
                        (c, e, share, share * (2.0 * rounding(tone.coefficients.len() + 5) + moved + rounding(3 * GROUND + 3)))
                    })
                    .collect();
                of.sort_by(|x, y| y.2.total_cmp(&x.2));
                let mut group = 0;
                (0..of.len())
                    .map(|i| {
                        if i > 0 && of[i - 1].2 - of[i].2 > of[i - 1].3 + of[i].3 {
                            group += 1;
                        }
                        Ranked { component: of[i].0, excited: of[i].1, group }
                    })
                    .collect()
            })
            .collect()
    }

    /// The model's beats: the distinct δ_ω − δ_ω' of its frequencies, each with its pairs (ω, ω') -- the first is zero,
    /// every frequency's with itself.
    pub(crate) fn beats_made(&self) -> Vec<(Beat, Vec<(usize, usize)>)> {
        let mut out: Vec<(Beat, Vec<(usize, usize)>)> = vec![];
        for to in 0..self.tones.len() {
            for from in 0..self.tones.len() {
                let beat = self.tones[to].beat.plus(-1, &self.tones[from].beat);
                match out.iter_mut().find(|b| b.0 == beat) {
                    Some(found) => found.1.push((to, from)),
                    None => out.push((beat, vec![(to, from)])),
                }
            }
        }
        out
    }

    /// Two frequencies' constant form (`Cross`), made once: a frequency's own is its static form.
    pub(crate) fn constants(&self, to: usize, from: usize) -> Arc<Cross> {
        let mut made = self.crosses.lock().expect("the constant forms' lock");
        made.entry((to, from))
            .or_insert_with(|| {
                let (t, f) = (&self.tones[to], &self.tones[from]);
                let forms = if to == from { t.forms.clone() } else { super::far::constant(t, f) };
                let pairs = if to == from { t.pairs.clone() } else { super::far::per_pair(t, f, &forms) };
                Arc::new(Cross { rounding: TermForm::rounding_of(self, to, from, &pairs), pairs, forms })
            })
            .clone()
    }

    /// The bound over a region of the slope of a state's level under the carriers (rad/µs/µm), made once per region and ground
    /// state: a ground state's shifts by the carriers (eliminate.rs `Model::shift_slope`). An explicit excited state's shift by
    /// the folded drives is not in it, as it is not in the couplings' distances (`Model::near_shift`).
    pub(crate) fn near_slope(&self, s: usize, region: &Region) -> f64 {
        if s < GROUND {
            *region.slopes[s].get_or_init(|| (0..self.carriers.len()).map(|c| self.shift_slope(c, s, region, false)).sum())
        } else {
            0.0
        }
    }

    /// The bounds over a region of the slopes |∇(E_e − E_b)| of the constant forms' levels (eliminate.rs
    /// `Model::energy_slope_over`) from ground state b to each excited state a folded drive reaches from it (per excited state
    /// less GROUND; the others' are not asked), made once per region and ground state: the carriers' light shifts' slopes and
    /// the drives' sweeps read them.
    pub(crate) fn level_slopes<'r>(&self, b: usize, region: &'r Region) -> &'r [f64; EXCITED] {
        region.levels[b].get_or_init(|| {
            let mut slopes = [0.0; EXCITED];
            for &e in &region.forms.reached(self)[b] {
                slopes[e] = self.energy_slope_over(GROUND + e, b, region);
            }
            slopes
        })
    }

    /// The bound over a region of an atom's acceleration (µm/µs²) in the clusters `within` picks: ħ/m times the force's bound
    /// (eliminate.rs `Model::force_over`).
    pub(crate) fn acceleration(&self, region: &Region, within: &dyn Fn(&Cluster) -> bool) -> f64 {
        self.hbar_over_mass * (0..self.clusters.len()).filter(|&c| within(&self.clusters[c])).map(|c| self.force_over(c, region)).fold(0.0, f64::max)
    }

    /// The gate over a region of a fold between any two states' levels -- a photon coherence's --, of size `size`, wave vector
    /// at most `wave` and a distance `distance` that is a bound over all space: the one gate across the distance less the
    /// Doppler shift q v, at the rate v L + q a_max -- L twice the largest slope of a state's level (the constant forms'
    /// energy, a ground state's shifts by the carriers), a_max the largest acceleration in any cluster, both made once per
    /// region.
    pub(crate) fn gate_over(&self, region: &Region, size: f64, wave: f64, distance: f64) -> Gate {
        let (slope, acceleration) = *region.moving.get_or_init(|| {
            let slope = (0..STATES).map(|s| self.energy_slope_over(s, s, region) + self.near_slope(s, region)).fold(0.0, f64::max);
            (2.0 * slope, self.acceleration(region, &|_| true))
        });
        gate(size, distance - wave * region.speed, region.speed * slope + wave * acceleration).resolved()
    }

    /// Whether a state is an explicit excited state of this model.
    pub(crate) fn explicit(&self, s: usize) -> bool {
        s >= GROUND && self.clusters.iter().any(|c| c.states.contains(&s))
    }

    /// The units that fold a component's drives with the bare levels (`Unit::Bare`), whatever their level in this model:
    /// kept, each of them is folded with the levels at the atom.
    pub fn bare_units(&self, component: Component) -> Vec<Unit> {
        self.driven.iter().filter(|d| d.0 == component).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect()
    }

    /// The candidate units of this model (model.rs `Unit`), those it folds: each folded drive at its level; the second
    /// order of each cluster's fold at the atom that has a fourth (model.rs `Model::fold_frequency`); the couplings
    /// of two ground states, and of an explicit excited state with another excited one, at each of the model's beats
    /// (module documentation, # A coupling of two states) that no cluster holds as they are; the carriers' dressings and the
    /// photon coherences (eliminate.rs, photon.rs).
    pub fn units(&self) -> Vec<Unit> {
        let mut out = vec![];
        for &(component, excited, ground) in &self.driven {
            match self.level(component, excited, ground) {
                Level::Bare => out.push(Unit::Bare { component, excited, ground }),
                Level::Atom => out.push(Unit::Drive { component, excited, ground }),
                Level::Explicit => {}
            }
        }
        // The second order of a cluster's fold at the atom, where the fold has a next order and is not taken to it.
        for (c, cluster) in self.clusters.iter().enumerate() {
            if self.fourth[c].is_none() && self.fold_frequency(c).is_some() {
                out.push(Unit::Second { ground: cluster.states[0] });
            }
        }
        let mut listed: HashSet<Unit> = HashSet::new();
        for (index, (beat, _)) in self.beats.iter().enumerate() {
            for a in 0..STATES {
                for b in 0..STATES {
                    if !(if a < GROUND { b < GROUND } else { b >= GROUND && (self.explicit(a) || self.explicit(b)) }) {
                        continue;
                    }
                    let unit = term_unit(a, b, beat);
                    if !listed.contains(&unit) && !self.forms.term(self, a, b, index).is_empty() && !self.in_generator(a, b, beat) {
                        listed.insert(unit.clone());
                        out.push(unit);
                    }
                }
            }
        }
        // A carrier's dressing: where its T̄ averages an interference (several polarization units) and an elimination takes
        // it -- some carrier has a drive folded at the atom, whose denominators hold its static form and its light shifts
        // (eliminate.rs, # The response).
        let eliminated = self.carriers.iter().any(|cc| cc.folded.iter().any(|x| x.data.iter().any(|z| *z != C64::ZERO)));
        for cc in self.carriers.iter().filter(|_| eliminated) {
            let mut units = cc.units.clone();
            units.sort();
            units.dedup();
            if units.len() > 1 {
                out.push(Unit::Dressing { carrier: Component { frequency: cc.frequency, counter: false } });
            }
        }
        out.extend(self.coherence_units());
        out.retain(|u| !self.kept.contains(u));
        out
    }

    /// Whether the coupling a ← b at a beat is in this model's generators as it is (SPEC §1.4), nothing of it folded: the
    /// two states in one cluster, at a residual beat of exactly zero or as a kept unit (eliminate.rs `Model::term_included`).
    fn in_generator(&self, a: usize, b: usize, beat: &Beat) -> bool {
        self.clusters.iter().enumerate().any(|(k, c)| {
            let place = |s: usize| c.states.iter().position(|&x| x == s);
            matches!((place(a), place(b)), (Some(x), Some(y)) if self.term_included(k, x, y, beat))
        })
    }

    /// The kept units' edges (cluster.rs): a kept drive at the atom joins its two states at θ_e − θ_g = δ_ω, its strongest
    /// beam's wave vector; a kept coupling a ← b at the beat β joins them at θ_a − θ_b = −β, its strongest beam pair's
    /// k_absorbed − k_emitted, its strength the peak of its entry with every drive bare, Σ_pairs |C_ji[a, b]| (`bare`: the
    /// frequencies with every drive folded with the bare levels -- one measure for every coupling, whatever the drives'
    /// levels turn out; the tree is a choice of cost, cluster.rs). A unit of nothing in this request makes none.
    pub(crate) fn edges(&self, bare: &[super::far::Tone]) -> Vec<Edge> {
        let mut out = vec![];
        let mut crosses: HashMap<(usize, usize), Vec<Mat>> = HashMap::new();
        for unit in &self.kept {
            match unit {
                Unit::Drive { component, excited, ground } if !component.counter && self.driven.contains(&(*component, *excited, *ground)) => {
                    let (f, tone) = (&self.light.frequencies[component.frequency], &self.tones[component.frequency]);
                    let (best, strength) = f.beams.iter().zip(&tone.couplings[0]).fold((f.beams[0], 0.0_f64), |(best, s), (&l, x)| {
                        let v = x[(*excited, *ground)].abs();
                        (if v > s { l } else { best }, s.max(v))
                    });
                    out.push(Edge { a: *excited, b: *ground, beat: tone.beat.clone(), wave: self.beam_wave(best), strength });
                }
                Unit::Term { a, b, beat } => {
                    let (mut best, mut strength) = ((0.0_f64, self.no_wave()), 0.0);
                    for &(to, from) in self.beats.iter().find(|x| x.0 == *beat).map_or(&[][..], |x| &x.1) {
                        let pairs = crosses.entry((to, from)).or_insert_with(|| {
                            if to == from { bare[to].pairs.clone() } else { super::far::per_pair(&bare[to], &bare[from], &super::far::constant(&bare[to], &bare[from])) }
                        });
                        let (bt, bf) = (&self.light.frequencies[to].beams, &self.light.frequencies[from].beams);
                        for (x, m) in pairs.iter().enumerate() {
                            let size = m[(*a, *b)].abs();
                            strength += size;
                            if size > best.0 {
                                best = (size, self.pair_wave(bt[x / bf.len()], bf[x % bf.len()]));
                            }
                        }
                    }
                    if strength > 0.0 {
                        out.push(Edge { a: *a, b: *b, beat: beat.negated(), wave: best.1, strength });
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// A ground state's drives over a region as their gates read them (`Across`; module documentation, # Levels), in the
    /// tail rule's order (an explicit one's unread); made once per region and ground state. A folded drive's size is
    /// v = |Σ_l a_l X_l[e, g]| (reach.rs `Region::coherent`); over the region its detuning lies within `moved` of its
    /// centre value Δ_c = (E_e − E_g)(c) − ν -- the two energies' slack and what the eliminations' levels average of them
    /// (eliminate.rs `CellEnergies`), the other components' carriers' dressing of g, the Doppler shift k·v of the region's
    /// speeds, the excited state's Gershgorin radius and the self-energy's bound (eliminate.rs), the folded channels'
    /// and the kept ones' (`kept_over`) --, so its distance there is at least |max(|Δ_c| − moved, 0) − iΓ_e/2|.
    pub(crate) fn across<'r>(&self, g: usize, region: &'r Region) -> &'r [Across] {
        region.across[g].get_or_init(|| {
            let cluster = self.membership[g];
            let (energies, couplings, dressings) = (self.energies(region), self.couplings(region), self.dressings(region));
            let folded = region.forms.folded(self);
            let dress: [f64; GROUND] = std::array::from_fn(|a| dressings.iter().map(|d| d[a]).sum());
            // Per component, what its drives of g share: the dressing of g by the other components' carriers, the largest
            // wave number of its beams, and the self-energy's bound per excited state -- its Gershgorin row,
            // Σ_{d, a∉C} |u_a[e]| ‖u_a‖/|z − Ẽ_a − δ_d| over the carriers' channels folded at the atom to the ground states of
            // the other clusters (only the states a channel reaches move).
            let mut shared: Vec<Option<(f64, f64, [f64; EXCITED])>> = vec![None; 2 * self.tones.len()];
            let kept = self.kept_over(region);
            region.forms.drives(self)[g]
                .iter()
                .map(|d| {
                    if d.level == Level::Explicit {
                        return Across { size: 0.0, centre: 0.0, moved: 0.0, still: 0.0 };
                    }
                    let (frequency, side) = (d.component / 2, d.component % 2);
                    let nu = self.tones[frequency].nu[side];
                    let (others, wave, sigma) = shared[d.component].get_or_insert_with(|| {
                        let others: f64 = self.carriers.iter().zip(dressings).filter(|of| self.other(of.0, d.component)).map(|of| of.1[g]).sum();
                        let (z_lo, z_hi) = (energies.centre[g] - energies.slack[g] + nu - others, energies.centre[g] + energies.slack[g] + nu + others);
                        let mut sigma = [0.0; EXCITED];
                        for ((c, cd), of) in self.carriers.iter().zip(couplings).zip(folded) {
                            for o in (0..GROUND).filter(|&o| self.membership[o] != cluster) {
                                let (lo, hi) = (z_lo - energies.centre[o] - energies.slack[o] - dress[o] - c.detuning, z_hi - energies.centre[o] + energies.slack[o] + dress[o] - c.detuning);
                                let (den, size) = (distance(lo, hi), cd.column[o]);
                                if size == 0.0 {
                                    continue;
                                }
                                for &x in &of.reached[o] {
                                    let reach = cd.entry[o][x];
                                    if reach > 0.0 {
                                        sigma[x] += reach * size / den;
                                    }
                                }
                            }
                        }
                        // The kept channels' rows: |V_a[e]| Σ_e' |V_a[e']| over the two-photon distance, per kept drive
                        // whose channel is another Floquet state than this component's from g.
                        let from = self.frequency(Component { frequency, counter: side == 1 });
                        for leg in kept.iter().filter(|leg| self.apart(g, &from, leg.ground, &self.carriers[leg.carrier].beat)) {
                            let (o, delta) = (leg.ground, self.carriers[leg.carrier].detuning);
                            let (lo, hi) = (z_lo - energies.centre[o] - energies.slack[o] - dress[o] - delta, z_hi - energies.centre[o] + energies.slack[o] + dress[o] - delta);
                            let row: f64 = kept.iter().filter(|other| other.carrier == leg.carrier && other.ground == o).map(|other| other.size).sum();
                            sigma[leg.excited] += leg.size * row / distance(lo, hi);
                        }
                        (others, self.wave_of(d.component), sigma)
                    });
                    let (x, e) = (d.excited, GROUND + d.excited);
                    let moved = energies.slack[e] + energies.slack[g] + energies.averaged[e] + energies.averaged[g] + *others + *wave * region.speed + energies.radius[e] + sigma[x];
                    let still = energies.slack[e] + energies.slack[g] + energies.averaged[e] + energies.averaged[g] + *others + energies.radius[e] + sigma[x];
                    Across { size: region.coherent(&d.beams), centre: energies.centre[e] - energies.centre[g] - nu, moved, still }
                })
                .collect()
        })
    }

    /// Whether a carrier is another component's than `component` (2 frequency + counter): its dressing of a ground state
    /// moves that component's drives (a carrier's own is its fold's).
    fn other(&self, carrier: &super::eliminate::Carrier, component: usize) -> bool {
        component % 2 == 1 || carrier.frequency != component / 2
    }

    /// The largest wave number of a component's beams (1/µm).
    fn wave_of(&self, component: usize) -> f64 {
        self.light.frequencies[component / 2].beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max)
    }

    /// The bound over a region of the shift of a ground state's level by its drives folded at the atom (rad/µs; module
    /// documentation, # What a folded drive does to its level): Σ v²/distance, each drive's coupling across its own
    /// distance there (`across`); made once per region and state. Nothing for an excited state: an explicit one's shift by
    /// the folded drives is in no distance.
    pub(crate) fn near_shift(&self, s: usize, region: &Region) -> f64 {
        if s >= GROUND {
            return 0.0;
        }
        *region.shifts[s].get_or_init(|| {
            let drives = &region.forms.drives(self)[s];
            drives.iter().zip(self.across(s, region)).filter(|of| of.0.level == Level::Atom).map(|(d, of)| of.size * of.size / of.distance(d.gamma)).sum()
        })
    }

    /// The least distance over a region of a carrier's drives of a ground state folded at the atom (`across`): what the
    /// carrier's response from that state is taken across, at least. None where it has no such drive.
    pub(crate) fn least_across(&self, carrier: usize, b: usize, region: &Region) -> Option<f64> {
        let component = 2 * self.carriers[carrier].frequency;
        let drives = &region.forms.drives(self)[b];
        let folded = drives.iter().zip(self.across(b, region)).filter(|of| of.0.level == Level::Atom && of.0.component == component);
        folded.map(|(d, of)| of.distance(d.gamma)).min_by(f64::total_cmp)
    }

    /// A ground state's drives' gates over a region (module documentation, # Levels, # The tail rule), per place of the tail
    /// rule's order: the gate of its drives at the atom -- the folded shares from there down, or the place's largest
    /// series gate where that is larger -- and the place's largest sweep; the same of its bare drives -- the shares from
    /// there down, the place's own share in each one's own gate's stead --; and the folded shares from there down, the
    /// tail. Every share is the region's own, the weaker places' as the place's: a point's are the values there. A folded
    /// drive's share is of its size across its distance over the region (`across`). A bare drive's own gate is of
    /// v + |Δ_c − Δ₀| + moved across its bare |Δ₀ − iΓ_e/2|; a drive at the atom's series gate of |w| across its distance
    /// at rest (`Across::distance_at_rest`), |w| ≤ k v_max + (ħ/m) k²/2 the Doppler shift and recoil of its photon, what its
    /// response takes to first order (eliminate.rs, # First-order Doppler). Its rate v L + k a_max, L the slope of E_e − E_g
    /// and of the other components' carriers' dressings of g. A drive onto an explicit state has the share and the sweep
    /// of its Raman pairs with that state's kept drives beside its own (`pair_gate`): its own statements, not shares in
    /// the tail -- a place's gate is the larger of its tail and its drives' largest pair.
    fn folds_made(&self, g: usize, region: &Region) -> Vec<[f64; 5]> {
        let drives = &region.forms.drives(self)[g];
        let kept = self.kept_over(region);
        let acceleration = self.hbar_over_mass * self.force_over(self.membership[g], region);
        let slopes = self.level_slopes(g, region);
        // Per component the slope of the other components' carriers' dressings of g and its beams' largest wave number.
        let mut shared: Vec<Option<(f64, f64)>> = vec![None; 2 * self.tones.len()];
        let mut out = vec![[0.0; 5]; drives.last().map_or(0, |d| d.group + 1)];
        // Per place its drives' largest pair saturation, at the atom and with the bare levels.
        let mut pairs = vec![[0.0_f64; 2]; out.len()];
        for (d, of) in drives.iter().zip(self.across(g, region)).filter(|of| of.0.level != Level::Explicit) {
            let (others, wave) = *shared[d.component].get_or_insert_with(|| {
                let others: f64 = (0..self.carriers.len()).filter(|&c| self.other(&self.carriers[c], d.component)).map(|c| self.shift_slope(c, g, region, true)).sum();
                (others, self.wave_of(d.component))
            });
            let rate = region.speed * (slopes[d.excited] + others) + wave * acceleration;
            let actual = gate(of.size, of.distance(d.gamma), rate);
            let pair = self.pair_gate(g, d, of, &kept, region);
            let place = &mut out[d.group];
            place[4] += actual.saturation;
            let bare = d.level == Level::Bare;
            pairs[d.group][bare as usize] = pairs[d.group][bare as usize].max(pair.saturation);
            if bare {
                let own = gate(of.size + (of.centre - d.detuning).abs() + of.moved, d.bare, rate);
                place[2] = place[2].max(own.saturation - actual.saturation);
                place[3] = place[3].max(own.resolved().sweep).max(pair.resolved().sweep);
            } else {
                let series = gate(wave * region.speed + 0.5 * self.hbar_over_mass * wave * wave, of.distance_at_rest(d.gamma), rate);
                place[0] = place[0].max(series.saturation);
                place[1] = place[1].max(actual.resolved().sweep).max(pair.resolved().sweep);
            }
        }
        // The tails: each place's shares with every weaker one's; a bare drive's own gate in its own share's stead, a drive
        // at the atom's series beside them. Then the place's pairs, each its own statement.
        for place in (0..out.len()).rev() {
            let tail = out[place][4] + out.get(place + 1).map_or(0.0, |weaker| weaker[4]);
            out[place][4] = tail;
            out[place][0] = out[place][0].max(tail);
            out[place][2] += tail;
        }
        for (place, pair) in out.iter_mut().zip(&pairs) {
            place[0] = place[0].max(pair[0]);
            place[2] = place[2].max(pair[1]);
        }
        out
    }

    /// The model's kept drives over a region (`Kept`), nothing where it has none.
    pub(crate) fn kept_over(&self, region: &Region) -> Vec<Kept> {
        if self.legs.iter().all(|of| of.is_empty()) {
            return vec![];
        }
        let energies = self.energies(region);
        let mut out = vec![];
        for (a, drives) in region.forms.drives(self).iter().enumerate() {
            for d in drives.iter().filter(|d| d.level == Level::Explicit) {
                let (frequency, e) = (d.component / 2, GROUND + d.excited);
                let carrier = self.carrier_of(frequency).expect("an explicit drive's component is a carrier");
                let wave = self.wave_of(d.component);
                let moved = energies.slack[e] + energies.slack[a] + wave * region.speed + energies.radius[e];
                let still = energies.slack[e] + energies.slack[a] + energies.radius[e];
                let across = Across { size: region.coherent(&d.beams), centre: energies.centre[e] - energies.centre[a] - self.carriers[carrier].detuning, moved, still };
                out.push(Kept { carrier, excited: d.excited, ground: a, size: across.size, distance: across.distance(d.gamma) });
            }
        }
        out
    }

    /// The gate over a region of the Raman pairs that a folded drive of ground state g onto an explicit state makes with
    /// that state's kept drives (module documentation, # Levels; far.rs, # A fold onto an explicit state): per kept
    /// drive V_a (e ← a at ν_a) whose channel is another Floquet state than the drive's (eliminate.rs `Model::apart`),
    /// the one gate of |U| |V_a| over the nearer of the two one-photon distances, across the two-photon distance
    /// |E_g + ν − E_a − ν_a| over the region -- its centre's, within how far the difference E_g − E_a moves over the
    /// region (the constant forms' diagonal difference, as a coupling's distance takes it: a shift the two levels share
    /// moves no distance) -- less the two-photon Doppler shift |q − k'| v (`two_photon_wave`), the shifts
    /// of the two ground levels by their drives folded at the atom (`near_shift`) and by their kept drives -- a kept drive
    /// of coupling v across the distance d moves its ground level by at most v and at most v²/d --, at the rate
    /// v |∇(E_g − E_a)| + |q − k'| a_max. The pairs' saturations summed (the drive's couplings to e's cluster), their
    /// largest sweep; nothing where the drive's excited state has no kept drive.
    fn pair_gate(&self, g: usize, d: &super::reach::DriveForm, of: &Across, kept: &[Kept], region: &Region) -> Gate {
        let mut out = Gate { saturation: 0.0, sweep: 0.0 };
        if kept.iter().all(|leg| leg.excited != d.excited) {
            return out;
        }
        let energies = self.energies(region);
        let (frequency, side) = (d.component / 2, d.component % 2);
        let (from, nu) = (self.frequency(Component { frequency, counter: side == 1 }), self.tones[frequency].nu[side]);
        let lifted = |s: usize| -> f64 { kept.iter().filter(|leg| leg.ground == s).map(|leg| leg.size.min(leg.size * leg.size / leg.distance)).sum() };
        for leg in kept.iter().filter(|leg| leg.excited == d.excited && self.apart(g, &from, leg.ground, &self.carriers[leg.carrier].beat)) {
            let a = leg.ground;
            let size = of.size * leg.size / of.distance(d.gamma).min(leg.distance);
            let wave = self.two_photon_wave(d, self.carriers[leg.carrier].frequency);
            let at = energies.centre[g] + nu - energies.centre[a] - self.carriers[leg.carrier].detuning;
            let slack: f64 = region.forms.difference(self, g, a).iter().map(|&(frequency, pair, _, modulus)| modulus * region.pairs[frequency][pair].radius).sum();
            let apart = distance(at - slack, at + slack) - wave * region.speed - self.near_shift(g, region) - self.near_shift(a, region) - lifted(g) - lifted(a);
            let slope = self.energy_slope_over(g, a, region) + self.near_slope(g, region) + self.near_slope(a, region);
            let acceleration = self.acceleration(region, &|c| c.states.contains(&g) || c.states.contains(&a));
            let pair = gate(size, apart, region.speed * slope + wave * acceleration);
            out.saturation += pair.saturation;
            out.sweep = out.sweep.max(pair.sweep);
        }
        out
    }

    /// The largest wave number of a Raman pair's two photons together (1/µm; module documentation, # Levels): |q − k'|
    /// over the beams of a drive folded onto an explicit state -- q = k_l, a counter-rotating component's −k_l -- and the
    /// beams k' of its kept leg's frequency, the pair's coupling turning in space as e^{i(q − k')·r}.
    fn two_photon_wave(&self, d: &super::reach::DriveForm, frequency: usize) -> f64 {
        let sign = if d.component % 2 == 1 { -1.0 } else { 1.0 };
        let mut most = 0.0_f64;
        for &(l, _, _) in &d.beams {
            let q = self.light.beams[l].wave_vector();
            for &m in &self.light.frequencies[frequency].beams {
                let k = self.light.beams[m].wave_vector();
                most = most.max(norm3(std::array::from_fn(|i| sign * q[i] - k[i])));
            }
        }
        most
    }

    /// A ground state's drives' gate at a place of the tail rule's order over a region (`folds_made`): [saturation, sweep]
    /// of its drives at the atom, then of its bare ones, then the tail. The region keeps the order's first `places` places -- those a
    /// proof over many cells still asks (`Model::gates`); every place's value is its own on the region whatever is
    /// kept --, and a place beyond them is made again.
    pub(crate) fn folds(&self, g: usize, place: usize, region: &Region, places: usize) -> [f64; 5] {
        let kept = region.drives[g].get_or_init(|| {
            let mut made = self.folds_made(g, region);
            made.truncate(places);
            made.shrink_to_fit();
            made
        });
        kept.get(place).copied().unwrap_or_else(|| self.folds_made(g, region)[place])
    }

    /// The bound over a region of the denominator of a drive e ← s folded at the atom at the detuning δ: its two levels'
    /// distance there, |E_e − E_s − δ| at the centre less their slack, what the eliminations' levels average of them and
    /// the excited state's Gershgorin radius (eliminate.rs `CellEnergies`), across Γ_e/2.
    pub(crate) fn denominator_over(&self, region: &Region, e: usize, s: usize, detuning: f64) -> f64 {
        let energies = self.energies(region);
        let apart = (energies.centre[e] - energies.centre[s] - detuning).abs() - energies.slack[e] - energies.slack[s] - energies.averaged[e] - energies.averaged[s] - energies.radius[e];
        C64::new(apart.max(0.0), 0.5 * self.decay[(e - GROUND, e - GROUND)].re).abs()
    }

    /// A coupling's (`Unit::Term`: a ← b at the model's beat `beat`) saturation and sweep over a region (module documentation,
    /// # A coupling of two states), made once per region. Its size: the constant part's value |H_ab + Σ_ji c_ji C_ji[a, b]| at
    /// the centre (H_ab the two states' own coupling with every light off, a static one's) plus Σ |C_ji[a, b]| times each
    /// pair's radius (reach.rs `PairSpan`), at most every beam at its largest amplitude and in phase; and the part at the atom, per excited state ½ |u'_a[e]| |u_b[e]| (1/σ_eb + 1/σ'_ea), u the two frequencies'
    /// folded couplings' coherent bounds and σ the denominator's bound of each drive folded at the atom (the bare drives'
    /// halves are the constants'). Its distance: |β + E_a − E_b| over the region -- the constant forms' difference, the two
    /// states' own --, less the Doppler shift of its beam pairs' largest wave vector q at the region's speeds and the
    /// two states' shifts by their drives folded at the atom (`near_shift`); its rate v |∇(E_a − E_b)| + q a_max, the two states' constant energies
    /// and shifts by the carriers (one state's beat moves with neither), the acceleration in their clusters. On a point, its
    /// measured values there.
    pub(crate) fn term_bound(&self, form: &TermForm, a: usize, b: usize, beat: usize, region: &Region) -> (f64, f64) {
        region.term((a, b, beat), || {
            let (mut centre, mut moved, mut blind) = (form.base, 0.0, form.base.abs());
            for &(lj, li, v, size) in &form.pairs {
                let pair = pair_span(&region.spans, lj, li);
                centre += pair.centre * v;
                moved += size * pair.radius;
                blind += size * region.spans[lj].most * region.spans[li].most;
            }
            let mut size = (centre.abs() + moved).min(blind);
            for leg in &form.legs {
                let e = GROUND + leg.excited;
                let halves = leg.from_atom.map_or(0.0, |detuning| 1.0 / self.denominator_over(region, e, b, detuning)) + leg.to_atom.map_or(0.0, |detuning| 1.0 / self.denominator_over(region, e, a, detuning));
                size += 0.5 * region.coherent(&leg.to) * region.coherent(&leg.from) * halves;
            }
            if size == 0.0 {
                return (0.0, 0.0);
            }
            let (mut at, mut slack) = (self.omega(&self.beats[beat].0), 0.0);
            if a != b {
                at += self.base[(a, a)].re - self.base[(b, b)].re;
                for &(frequency, pair, d, modulus) in region.forms.difference(self, a, b) {
                    let pair = &region.pairs[frequency][pair];
                    at += (pair.centre * d).re;
                    slack += modulus * pair.radius;
                }
            }
            let (spread, slope) = if a == b { (0.0, 0.0) } else { (self.near_spread(a, b, region), self.energy_slope_over(a, b, region) + self.near_slope(a, region) + self.near_slope(b, region)) };
            let acceleration = self.acceleration(region, &|c| c.states.contains(&a) || c.states.contains(&b));
            let made = gate(size, distance(at - slack, at + slack) - form.wave * region.speed - spread, region.speed * slope + form.wave * acceleration).resolved();
            (made.saturation, made.sweep)
        })
    }

    /// The model's beat a unit's coupling names, its place among `Model::beats`.
    fn beat_index(&self, beat: &Beat) -> Option<usize> {
        self.beats.iter().position(|b| b.0 == *beat)
    }

    /// What each unit's gate reads (`Judged`), made once for a model's units: a drive's its ground state, its place in the
    /// tail rule's order and whether it is bare; a coupling's its form; a coherence's its unit and its distance over all
    /// space; nothing for a dressing, which folds nothing, a fold's order -- where the fold exists, so does its next order
    /// -- and a unit of nothing in this model.
    pub(crate) fn judged<'u>(&self, units: &'u [Unit]) -> Judged<'u> {
        let reads = units
            .iter()
            .map(|u| match *u {
                Unit::Bare { component, excited, ground } | Unit::Drive { component, excited, ground } => {
                    let level = self.ranked[ground].iter().find(|d| d.component == component && d.excited == excited).map(|d| (d.group, self.level(component, excited, ground)));
                    level.filter(|of| of.1 != Level::Explicit).map_or(Read::Nothing, |(place, level)| Read::Fold { ground, place, bare: level == Level::Bare })
                }
                Unit::Term { a, b, ref beat } => self.beat_index(beat).map_or(Read::Nothing, |index| Read::Term { form: self.forms.term(self, a, b, index), a, b, beat: index }),
                Unit::Dressing { .. } | Unit::Second { .. } => Read::Nothing,
                Unit::Coherence { ref first, ref second, .. } => Read::Coherence { unit: u, apart: self.omega(&first.plus(-1, second)).abs() - self.spread() },
            })
            .collect();
        Judged { reads, wave: 2.0 * self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max) }
    }

    /// A unit's gate over a region, [saturation, sweep] (module documentation, # One rule): its read (`Judged`) of the
    /// region -- a cell of a reach's proof, the whole space, or a step's path (solve.rs `Domain`): one statement, its value
    /// on a point the point's own --, every kind through the one gate (`gate`): a drive's by its ground state's tail
    /// (`folds`, the region keeping its first `places` places), a coupling's (`term_bound`), a photon coherence's
    /// (`gate_over`, its size its two classes' jumps' bound, photon.rs).
    pub(crate) fn gate_of(&self, judged: &Judged, u: usize, region: &Region, places: usize) -> [f64; 2] {
        match &judged.reads[u] {
            Read::Fold { ground, place, bare } => {
                let made = self.folds(*ground, *place, region, places);
                if *bare { [made[2], made[3]] } else { [made[0], made[1]] }
            }
            Read::Term { form, a, b, beat } => {
                let made = self.term_bound(form, *a, *b, *beat, region);
                [made.0, made.1]
            }
            Read::Coherence { unit, apart } => {
                let made = self.gate_over(region, self.coherence_bound(unit, region), judged.wave, *apart);
                [made.saturation, made.sweep]
            }
            Read::Nothing => [0.0; 2],
        }
    }

    /// Each unit's gate over the whole space at the speeds up to `speed`, every beam at its peak (`gate_of`): a bound over any
    /// region at those speeds; and the units it leaves open -- some bound at ½ or above --, the others' gates.
    pub(crate) fn everywhere(&self, judged: &Judged, speed: f64) -> (Vec<[f64; 2]>, Vec<usize>) {
        let whole = self.region(self.spans(&Cell { centre: [0.0; 3], half: [f64::INFINITY; 3] }), speed);
        let everywhere: Vec<[f64; 2]> = (0..judged.reads.len()).map(|u| self.gate_of(judged, u, &whole, usize::MAX)).collect();
        let open = (0..everywhere.len()).filter(|&u| !(everywhere[u][0] < EXISTS && everywhere[u][1] < EXISTS)).collect();
        (everywhere, open)
    }

    /// Whether every one of the units `which` has its fold over a region (`gate_of`, `Gate::exists`).
    pub(crate) fn exist_over(&self, judged: &Judged, which: &[usize], region: &Region) -> bool {
        which.iter().all(|&u| {
            let [saturation, sweep] = self.gate_of(judged, u, region, usize::MAX);
            saturation < EXISTS && sweep < EXISTS
        })
    }

    /// Each unit's existence gate over the reach (SPEC §1.5; module documentation), provable bounds at this model's clusters
    /// and levels (`gate_of`). Each is bounded on the whole space first, every beam at its peak (`everywhere`): a bound over
    /// any reach, and where both of a unit's are below ½ there its gate. The others are proven together over the reach's
    /// cells, on all the cores (reach.rs `Model::bounds_over_in_parallel`) -- a ground state's drive's with every weaker
    /// one's share on the same cell: what a cell's centre attains is a value there, never a bound from elsewhere --, each
    /// until it is below ½ everywhere or a centre attains `KEEP` (module documentation, # One rule).
    pub fn gates(&self, units: &[Unit], reach: &Reach) -> Vec<Gate> {
        let judged = self.judged(units);
        let (everywhere, open) = self.everywhere(&judged, reach.speed);
        // Per ground state the places its open drives' gates ask of a cell.
        let mut places = [0; GROUND];
        for &u in &open {
            if let Read::Fold { ground, place, .. } = judged.reads[u] {
                places[ground] = places[ground].max(place + 1);
            }
        }
        let mut quantities: Vec<Box<dyn Fn(&Region) -> f64 + Sync + '_>> = vec![];
        let mut shared: HashMap<(usize, usize, bool), usize> = HashMap::new();
        // Per open unit where its two bounds lie among the quantities: the saturation's, then the sweep's.
        let judged = &judged;
        let asked: Vec<usize> = open
            .iter()
            .map(|&u| {
                let (key, asks) = if let Read::Fold { ground, place, bare } = judged.reads[u] { (Some((ground, place, bare)), places[ground]) } else { (None, 0) };
                if let Some(found) = key.and_then(|k| shared.get(&k)) {
                    return *found;
                }
                let first = quantities.len();
                quantities.push(Box::new(move |r: &Region| self.gate_of(judged, u, r, asks)[0]));
                quantities.push(Box::new(move |r: &Region| self.gate_of(judged, u, r, asks)[1]));
                if let Some(k) = key {
                    shared.insert(k, first);
                }
                first
            })
            .collect();
        let refs: Vec<&(dyn Fn(&Region) -> f64 + Sync)> = quantities.iter().map(|q| q.as_ref()).collect();
        let proven = self.bounds_over_in_parallel(reach, EXISTS, KEEP, &refs);
        let mut out: Vec<Gate> = everywhere.iter().map(|of| Gate { saturation: of[0], sweep: of[1] }).collect();
        for (&u, &at) in open.iter().zip(&asked) {
            out[u] = Gate { saturation: proven[at], sweep: proven[at + 1] };
        }
        out
    }

    /// Each candidate unit's errors at each sample (`Model::units`), [unit][sample]: its small parameter at the sample's point
    /// mapped to the outputs (module documentation, # The errors), the forms at the atom from the exact elimination at the
    /// point.
    pub fn errors(&self, units: &[Unit], samples: &[Sample]) -> Vec<Vec<UnitError>> {
        self.errors_with(units, samples, None)
    }

    /// The units' errors at the samples (`errors`), the forms at the atom from their tables where they serve (tabulate.rs): a
    /// state then costs an interpolation and the units' arithmetic, and one elimination more per dressing unit.
    pub fn errors_with(&self, units: &[Unit], samples: &[Sample], tables: Option<&Tables>) -> Vec<Vec<UnitError>> {
        let mut out: Vec<Vec<UnitError>> = units.iter().map(|_| Vec::with_capacity(samples.len())).collect();
        for sample in samples {
            for (errors, error) in out.iter_mut().zip(self.errors_at(units, sample, tables)) {
                errors.push(error);
            }
        }
        out
    }

    /// The units' errors at one sample (`errors_with`), in the units' order: what the sample's units share made once
    /// (`Measured`), then each unit's arithmetic.
    pub fn errors_at(&self, units: &[Unit], sample: &Sample, tables: Option<&Tables>) -> Vec<UnitError> {
        let at = self.measured(sample, tables);
        units.iter().map(|unit| self.unit_error(unit, &at)).collect()
    }

    /// What the units' errors at a sample share (`Measured`).
    pub(crate) fn measured<'a>(&self, sample: &'a Sample, tables: Option<&Tables>) -> Measured<'a> {
        let lights = self.running();
        let amplitudes = self.light.amplitudes(sample.at.r);
        let plan = self.plan(sample.cluster, &lights);
        let sections = [(plan.near.hermitian(), 1), (plan.near.jumping(), 0), (plan.near.responding(), 0)];
        let near = self.near_values(&plan, &amplitudes, &sections, tables.and_then(|t| t.of(sample.cluster, &lights)));
        let responses = self.carrier_responses(&plan, &near, &amplitudes, sample.at.kappa);
        // The force on the state at the point: −tr(G ρ) of its density (step.rs `Middle::force`, at rest on the middle).
        let middle = self.middle_of(&plan, sample.at.t, &amplitudes, Some(&near));
        let (n, norm) = (sample.psi.len(), crate::linalg::norm_sqr(&sample.psi));
        let rho: Vec<C64> = (0..n * n).map(|i| sample.psi[i / n] * sample.psi[i % n].conj() / norm).collect();
        let force = crate::step::Middle::force(&middle, &crate::step::Middle::path(&middle, sample.at.kappa, [0.0; 3]), 0.0, &rho).map(|f| -f);
        // The state's folded excitation drive by drive: a bare one's coupling is its response times its bare denominator, one
        // at the atom's is its beams' at the point.
        let bare = self.far_channels(sample.cluster, &amplitudes, &lights);
        let populations = self.state_populations(sample);
        let c = &self.clusters[sample.cluster];
        let (mut rates, mut shifts) = (0.0, 0.0);
        let mut folds = vec![vec![Fold { share: 0.0, level: C64::ZERO, terms: [0.0; EXCITED], across: OnceCell::new() }; c.ground]; self.carriers.len()];
        for (y, &g) in c.states[..c.ground].iter().enumerate() {
            for x in 0..EXCITED {
                for channel in &bare {
                    let size = channel.response[(x, y)].norm_sqr();
                    if size > 0.0 {
                        rates += size * populations[g];
                        shifts += size * self.bare_distance(channel.component, GROUND + x, g) * populations[g];
                    }
                }
                for ((cc, response), fold) in self.carriers.iter().zip(&responses).zip(&mut folds) {
                    let response = response.as_ref().map_or(C64::ZERO, |r| r[(x, y)]);
                    let size = response.abs();
                    if size > 0.0 {
                        let coupling: C64 = cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + x, g)]).sum();
                        rates += size * size * populations[g];
                        shifts += size * coupling.abs() * populations[g];
                        // (A drive's size the modulus of its own term: one drive's is the level's, to the bit.)
                        let term = coupling.conj() * response;
                        fold[y].share += 2.0 * size * size;
                        fold[y].level += term;
                        fold[y].terms[x] = term.abs();
                    }
                }
            }
        }
        // The kept drives' part of both: an explicit state's population is its excitation as it is, and its coupling's
        // energy |V[e, g]| |ψ_e| |ψ_g| what a fold's |V R| p_g is of ψ_e = −R ψ_g.
        for &e in &c.states[c.ground..] {
            rates += populations[e];
            for &g in &c.states[..c.ground] {
                let coupling: C64 = self
                    .carriers
                    .iter()
                    .filter(|cc| self.level(Component { frequency: cc.frequency, counter: false }, e, g) == Level::Explicit)
                    .flat_map(|cc| cc.beams.iter().zip(&cc.couplings).map(|(&l, m)| amplitudes[l].value * m[(e, g)]))
                    .sum();
                shifts += coupling.abs() * (populations[e] * populations[g]).sqrt();
            }
        }
        Measured {
            sample,
            region: self.point_region(sample.at.r, self.hbar_over_mass * norm3(sample.at.kappa)),
            near,
            responses,
            bare,
            rates,
            shifts,
            folds,
            energies: {
                let mut energies = vec![];
                self.far_energies(&amplitudes, &lights, 1, &self.coherent_dressing(), &mut energies);
                energies
            },
            force,
            populations,
            classes: OnceCell::new(),
            doppler: self.carriers.iter().map(|_| OnceCell::new()).collect(),
            kept: OnceCell::new(),
            lifts: OnceCell::new(),
            amplitudes,
            plan,
        }
    }

    /// A unit's error at a sample from what the sample's units share.
    fn unit_error(&self, unit: &Unit, at: &Measured) -> UnitError {
        match unit {
            Unit::Bare { component, excited, ground } | Unit::Drive { component, excited, ground } => {
                let (own, response) = if matches!(unit, Unit::Bare { .. }) { self.bare_error(unit, at) } else { self.drive_error(unit, at) };
                let pair = self.pair_error(*component, *excited, *ground, response, at);
                UnitError { parameter: own.parameter.max(pair.parameter), population: own.population + pair.population, temperature: own.temperature + pair.temperature }
            }
            Unit::Second { ground } => self.order_error(*ground, at),
            Unit::Term { a, b, beat } => {
                let m2 = self.beat_index(beat).map_or(0.0, |index| 0.5 * self.term_bound(&self.forms.term(self, *a, *b, index), *a, *b, index, &at.region).0);
                let p = &at.populations;
                UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 * (p[*a] + p[*b]) }
            }
            Unit::Dressing { carrier } => self.dressing_error(*carrier, at),
            Unit::Coherence { .. } => self.coherence_error(unit, at),
        }
    }

    /// The error at a sample of the second order of a cluster's fold at the atom (`Unit::Second`; module documentation,
    /// # The errors, # The fold's order): Σ s M/S_g p_g over the drives folded at the atom from the sample's ground
    /// states -- s = 2|R[e, g]|² each one's share, M the larger of its ground state's share S_g and of that level's own
    /// term across the drive's distance (eliminate.rs `Folded::order`; at one distance Σ s p_g) -- on the populations,
    /// the rates' relative error of that order; on T the same, which bounds the force's (Σ s p_g at any distances).
    /// Zero off the cluster, and where its fold is the fourth order's.
    fn order_error(&self, ground: usize, at: &Measured) -> UnitError {
        let cluster = self.membership[ground];
        let Some(frequency) = self.fold_frequency(cluster).filter(|_| at.sample.cluster == cluster && self.fourth[cluster].is_none()) else {
            return UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        };
        let component = Component { frequency, counter: false };
        let c = &self.clusters[cluster];
        let (mut sum, mut held) = (0.0, 0.0);
        if self.carrier_of(frequency).is_some_and(|ci| at.responses[ci].is_some()) {
            for &g in &c.states {
                held += at.populations[g];
                for e in (GROUND..STATES).filter(|&e| self.level(component, e, g) == Level::Atom) {
                    if let Some(drive) = self.drive_parameters(&Unit::Drive { component, excited: e, ground: g }, at) {
                        sum += drive.order(1) * drive.population;
                    }
                }
            }
        }
        UnitError { parameter: if held > 0.0 { sum / held } else { 0.0 }, population: sum, temperature: sum }
    }

    /// The error at a sample of a drive folded with the bare levels (module documentation, # Levels, # The errors): its own
    /// share s = 2|R⁰[e, g]|² of its ground state, as every fold's -- s p_g on the populations, (s + a) p_g on T, a its
    /// sweep across its denominator at the atom --; and the first order of its denominator, ε = |δΔ|/|Δ₀ − iΓ_e/2|
    /// (eliminate.rs `Model::drive_motion`), the relative error of its response: of the state's rates, quadratic in the
    /// responses, 2ε times its share |R⁰[e, g]|² p_g of them, and of its light shifts, linear, ε times its share
    /// |V R⁰| p_g of those (`Measured::rates`, `Measured::shifts`: shares that add up to one over the state's drives,
    /// folded and kept, so that the units' errors add up to the state's) -- the rates' on the populations, both on T, booked on
    /// the state's ground population as a dressing's relative rate error is. Zero where the sample's cluster does not
    /// hold its ground state. With the drive's response there, where the cluster holds it (the Raman pairs' error reads
    /// it, eliminate.rs `Model::pair_error`).
    fn bare_error(&self, unit: &Unit, at: &Measured) -> (UnitError, Option<C64>) {
        let nothing = (UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 }, None);
        let Unit::Bare { component, excited, ground } = *unit else { return nothing };
        let c = &self.clusters[at.sample.cluster];
        let (Some(y), Some(channel)) = (c.states[..c.ground].iter().position(|&s| s == ground), at.bare.iter().find(|channel| channel.component == component)) else { return nothing };
        let response = channel.response[(excited - GROUND, y)];
        let s = 2.0 * response.norm_sqr();
        let (moved, rate) = self.drive_motion(component, excited, ground, at);
        let own = self.bare_distance(component, excited, ground);
        let gamma = self.decay[(excited - GROUND, excited - GROUND)].re;
        let nu = self.tones[component.frequency].nu[component.counter as usize];
        let sweep = gate(0.0, C64::new(self.energies[excited] - self.energies[ground] - nu + moved, 0.5 * gamma).abs(), rate).sweep;
        let (epsilon, p, held) = (moved.abs() / own, at.populations[ground], at.populations[..GROUND].iter().sum::<f64>());
        let share = |of: f64, all: f64| if all > 0.0 { of * p / all } else { 0.0 };
        let (rates, shifts) = (2.0 * epsilon * share(response.norm_sqr(), at.rates) * held, epsilon * share(response.norm_sqr() * own, at.shifts) * held);
        (UnitError { parameter: epsilon, population: s * p + rates, temperature: (s + sweep) * p + rates + shifts }, Some(response))
    }

    /// A sample's populations of the model basis' states on its cluster (|ψ_s|²/‖ψ‖²; zero off its cluster).
    pub(crate) fn state_populations(&self, s: &Sample) -> [f64; STATES] {
        let c = &self.clusters[s.cluster];
        let total: f64 = s.psi.iter().map(|z| z.norm_sqr()).sum();
        let mut p = [0.0; STATES];
        for (i, &x) in c.states.iter().enumerate() {
            p[x] = s.psi[i].norm_sqr() / total;
        }
        p
    }

    /// The exact wave vector of a beam pair's term, absorbing `absorbed` and emitting `emitted`: k_absorbed − k_emitted.
    pub(crate) fn pair_wave(&self, emitted: usize, absorbed: usize) -> Wave {
        self.beam_wave(absorbed).plus(-1, &self.beam_wave(emitted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::{Point, Sample};
    use crate::request::Request;

    /// A scene's model (a reference scene by name, or a native request by its path) closed over its request's reach (solve.rs `attempt`: every unit whose gate fails kept).
    fn closed(scene: &str) -> (Model, [f64; 3]) {
        let path = if scene.ends_with(".json") { std::path::PathBuf::from(scene) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json")) };
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let state = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let mut kept: Vec<Unit> = vec![];
        loop {
            let model = Model::new(&request, &light, &kept).unwrap();
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let reach = model.reach(&trap, &reference, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
            let units = model.units();
            let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
            if failing.is_empty() {
                return (model, trap.centre);
            }
            kept.extend(failing);
        }
    }

    /// The sweep is a relative error of the fold's own effect (module documentation, # One rule; `Gate::resolved`). The
    /// exact response of a linear sweep is the instantaneous fold times I(a) = ∫₀^∞ e^{−x + iax²/2} dx (internal.md
    /// 1B.1), of modulus at most one at any a -- so the fold is off by at most twice its own amplitude: |I(a) − 1| ≤ 2
    /// (Simpson on [0, 40], the step resolving the phase's largest rate a·40). At a = ½ the response holds 0.71 of the
    /// fold's population, at a = 1.75 0.39 (the gate's edge, internal.md). A coupling whose whole effect, 2S, is below the
    /// least resolved share has its sweep scaled down with it -- the user's lin⊥lin pair, S = 9e-38 at a sweep of 0.83,
    /// has none --; one above it keeps its sweep, and a fold of a share at the target's scale with a sweep beyond ½ is
    /// kept. Continuous across the resolution (a cut-off there left a cell's bound at 8.4 over its centre's 0).
    #[test]
    fn a_fold_s_sweep_counts_as_far_as_its_own_effect_is_seen() {
        let response = |a: f64| -> C64 {
            let (end, n) = (40.0_f64, (2.0 * (1.0 + a * 40.0) * 40.0 * 20.0).ceil() as usize * 2);
            let h = end / n as f64;
            let f = |x: f64| C64::new(-x, 0.5 * a * x * x).exp();
            let inner: C64 = (1..n).map(|i| f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }).sum();
            (f(0.0) + inner + f(end)) * (h / 3.0)
        };
        for a in [0.0, 0.1, 0.5, 1.0, 1.75, 4.0, 20.0] {
            let i = response(a);
            assert!(i.abs() <= 1.0 + 1e-9 && (i - C64::ONE).abs() <= 2.0, "a = {a}: I = {i:?}");
        }
        assert!((response(0.5).norm_sqr() - 0.71).abs() < 0.01 && (response(1.75).norm_sqr() - 0.39).abs() < 0.01, "{} {}", response(0.5).norm_sqr(), response(1.75).norm_sqr());
        let unseen = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        assert!(Gate { saturation: 9e-38, sweep: 0.83 }.resolved().exists());
        assert!(Gate { saturation: 0.25 * unseen, sweep: 0.9 }.resolved().exists());
        assert!(!Gate { saturation: unseen, sweep: 0.6 }.resolved().exists());
        assert!(!Gate { saturation: 0.1, sweep: 0.6 }.resolved().exists());
        assert_eq!(Gate { saturation: 0.1, sweep: 0.3 }.resolved().sweep, 0.3);
        // Continuous across the resolution: just below and just above 2S = unseen the sweep is the same to the rounding.
        let (below, above) = (Gate { saturation: 0.5 * unseen * (1.0 - 1e-12), sweep: 8.4 }.resolved().sweep, Gate { saturation: 0.5 * unseen * (1.0 + 1e-12), sweep: 8.4 }.resolved().sweep);
        assert!((above - below).abs() <= 1e-10, "{below} {above}");
    }

    /// A gate's bound of its unit's error (`Gate::error_bound`) at states the reach holds. On the closed models of
    /// R6_RAMAN_CO (couplings at a Raman pair's beats, photon coherences) and R4_LIN_1 (a lin-perp-lin lattice's): 200
    /// atoms each drawn in the reach's box with a momentum of a speed up to the reach's and kept where the reach holds
    /// them (`Model::holds`); on every cluster a random state there and each of its basis states (the populations'
    /// extremes). Every bounded unit's error on the populations and on T is within its bound, to the rounding of the
    /// few operations that make both. The bound is attained: on R6_RAMAN_CO a level's beating shift at its own state
    /// (p_a = 1, the coupling's size the same at every point) reads more than half of it.
    #[test]
    fn a_gate_bounds_its_unit_s_error_at_the_states_the_reach_holds() {
        for (scene, attained) in [("R6_RAMAN_CO", 0.5), ("R4_LIN_1", 0.0)] {
            let (model, _) = closed(scene);
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json"));
            let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
            let state = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
            let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let reach = model.reach(&trap, &reference, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
            let units = model.units();
            let gates = model.gates(&units, &reach);
            let bounds: Vec<Option<f64>> = units.iter().zip(&gates).map(|(u, g)| g.error_bound(u)).collect();
            assert!(bounds.iter().flatten().count() > 0, "{scene}: no bounded unit");
            let mut stream = crate::random::Stream::new(0, crate::random::Purpose::Initial, 0, 0);
            let (mut held, mut largest) = (0, 0.0_f64);
            while held < 200 {
                let r: [f64; 3] = std::array::from_fn(|a| reach.cell[a][0] + (reach.cell[a][1] - reach.cell[a][0]) * stream.uniform());
                let (direction, speed) = (stream.normals(), reach.speed * stream.uniform());
                let kappa = direction.map(|x| x / norm3(direction) * speed / model.hbar_over_mass);
                if !model.holds(&reach, r, kappa) {
                    continue;
                }
                held += 1;
                for (cluster, c) in model.clusters.iter().enumerate() {
                    let n = c.states.len();
                    let random: Vec<C64> = (0..n).map(|_| { let z = stream.normals(); C64::new(z[0], z[1]) }).collect();
                    let basis = (0..n).map(|i| (0..n).map(|j| if i == j { C64::real(1.0) } else { C64::ZERO }).collect::<Vec<C64>>());
                    for psi in std::iter::once(random).chain(basis) {
                        let sample = Sample { cluster, at: Point { r, kappa, t: 0.0 }, psi };
                        for ((unit, error), bound) in units.iter().zip(model.errors_at(&units, &sample, None)).zip(&bounds) {
                            let Some(bound) = bound else { continue };
                            let most = error.population.max(error.temperature);
                            assert!(most <= bound * (1.0 + rounding(8)), "{scene}: {unit:?} reads {most:e} beyond its bound {bound:e} at {r:?}");
                            if *bound > 0.0 {
                                largest = largest.max(most / bound);
                            }
                        }
                    }
                }
            }
            assert!(largest > attained, "{scene}: the largest error over its bound {largest}");
        }
    }

    /// Three levels in a scene's atom: every coupling of R6_RAMAN_CO's request removed but g –V→ e of one frequency and
    /// b –U→ e of the other, no outside form, the levels set so that the kept drive is ε from e and the folded one β
    /// (in Γ_e). The request; the two frequencies; Γ_e.
    fn three_levels(v: f64, epsilon: f64, u: f64, beta: f64) -> (Request, [usize; 2], f64) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R6_RAMAN_CO.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let (g, b, e) = (THREE[0], THREE[1], THREE[2]);
        let of: Vec<usize> = (0..light.frequencies.len()).filter(|&w| w != light.trap).collect();
        assert!(of.len() == 2 && of.iter().all(|&w| light.frequencies[w].beams.len() == 1), "two lights of one beam each beside the trap's");
        let nu: Vec<f64> = { let model = Model::new(&request, &light, &[]).unwrap(); of.iter().map(|&w| model.tones[w].nu[0]).collect() };
        let gamma: f64 = request.emission.iter().map(|c| (0..GROUND).map(|s| c[(s, e)].norm_sqr()).sum::<f64>()).sum();
        for beam in request.beams.iter_mut() {
            beam.coupling = Mat::zeros(STATES, STATES);
            beam.counter_coupling = Mat::zeros(STATES, STATES);
        }
        request.beams[light.frequencies[of[0]].beams[0]].coupling[(e, g)] = C64::real(v * gamma);
        request.beams[light.frequencies[of[1]].beams[0]].coupling[(e, b)] = C64::real(u * gamma);
        for form in request.outside_forms.iter_mut() {
            form.matrices = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
        }
        request.hamiltonian[(g, g)] = C64::ZERO;
        request.hamiltonian[(e, e)] = C64::real(epsilon * gamma + nu[0]);
        request.hamiltonian[(b, b)] = C64::real(epsilon * gamma + nu[0] - nu[1] - beta * gamma);
        (request, [of[0], of[1]], gamma)
    }
    /// `three_levels`' states: g, b, e.
    const THREE: [usize; 3] = [3, 0, GROUND];

    /// A cluster's no-jump generator at the focus, at rest; its eigenvalue whose vector is mostly the place `of`, and
    /// that vector.
    fn eigenvalue(model: &Model, cluster: usize, of: usize) -> (C64, Vec<C64>) {
        use crate::step::Middle as _;
        let middle = model.assemble(cluster, [0.0; 3], 0.0, &model.running());
        let n = model.clusters[cluster].states.len();
        let mut h = Mat::zeros(n, n);
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
        like(&h, of)
    }
    fn like(h: &Mat, of: usize) -> (C64, Vec<C64>) {
        let decomposed = crate::linalg::eigen(h).unwrap();
        let k = (0..h.rows).max_by(|&x, &y| decomposed.vectors[(of, x)].norm_sqr().total_cmp(&decomposed.vectors[(of, y)].norm_sqr())).unwrap();
        (decomposed.values[k], (0..h.rows).map(|x| decomposed.vectors[(x, k)]).collect())
    }

    /// A drive folded onto an excited state that another drive makes explicit is a Raman pair (far.rs, # A fold onto
    /// an explicit state), against the exact three levels g –V→ e ←U– b (`three_levels`; the exact no-jump spectrum of
    /// the 3 × 3 generator: the kept cluster's own 2 × 2 without U, b at the two-photon distance d_R = β − ε from g,
    /// U between e and b). V = 0.5 Γ, ε = Γ, U = 0.2 Γ, β from 10 Γ through the Raman resonance at β = ε to 0.02 Γ, the
    /// folded drive with the bare levels and with those at the atom:
    /// - where the drive's gate holds at the point (`Model::folds`: its own and the pair's), the kept line's rate
    ///   −2 Im λ_g and b's rate −2 Im λ_b are the exact ones within the drive's booked error (`Model::errors`, at the
    ///   state itself; the populations' error over the state's ground population) -- b's, where its fold's order is
    ///   its cluster's unit (`Unit::Second`), with that unit's s times the 3/2 the second order's slow decay is off by
    ///   (SPEC §1.5; the photons' rate, which s is booked for, by s). Measured: the kept line's rate +1.7e-4 at 10 Γ
    ///   (booked 2.4e-4), +1.6e-2 at 2 Γ (2.8e-2), +1.2e-2 at 0.02 Γ (3.5e-2), where the one-photon real denominator
    ///   read 82 % low;
    /// - it holds far from the resonance and fails around it (bare: from 1.5 Γ down, its own gate's too -- b's
    ///   undressed response is 19 % off at 2 Γ --; at the atom: from 1.2 Γ to 0.1 Γ): the fold is refused there, not
    ///   evaluated.
    /// And b's response through an e that a strong drive dresses (V = 2 Γ on resonance, β = 27.4 Γ): with the levels at
    /// the atom its rate is the exact one to 1e-3 (the undressed response's: −1.06 %, within its booked error).
    #[test]
    fn a_drive_folded_onto_an_explicit_state_is_a_raman_pair() {
        let (g, b, e) = (THREE[0], THREE[1], THREE[2]);
        let case = |v: f64, epsilon: f64, u: f64, beta: f64, atom: bool| -> (bool, [f64; 2], [f64; 2]) {
            let (request, of, gamma) = three_levels(v, epsilon, u, beta);
            let light = Light::new(&request);
            let (kept, folded) = (Component { frequency: of[0], counter: false }, Component { frequency: of[1], counter: false });
            let explicit = Unit::Drive { component: kept, excited: e, ground: g };
            let bare = Unit::Bare { component: folded, excited: e, ground: b };
            let model = Model::new(&request, &light, &if atom { vec![explicit.clone(), bare.clone()] } else { vec![explicit.clone()] }).unwrap();
            let unit = if atom { Unit::Drive { component: folded, excited: e, ground: b } } else { bare };
            assert_eq!(model.level(folded, e, b), if atom { Level::Atom } else { Level::Bare });
            let (cg, cb) = (model.membership[g], model.membership[b]);
            assert_eq!((model.clusters[cg].states.clone(), model.clusters[cb].states.clone()), (vec![g, e], vec![b]));
            // The exact three levels: the kept cluster's own generator without the folded light, and b.
            let alone = {
                let (without, _, _) = three_levels(v, epsilon, 0.0, beta);
                let model = Model::new(&without, &Light::new(&without), &[explicit]).unwrap();
                use crate::step::Middle as _;
                let middle = model.assemble(model.membership[g], [0.0; 3], 0.0, &model.running());
                let mut h = Mat::zeros(2, 2);
                middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                h
            };
            let exact = Mat::from_fn(3, 3, |x, y| match (x, y) {
                (2, 2) => alone[(0, 0)] - C64::real((beta - epsilon) * gamma),
                (1, 2) | (2, 1) => C64::real(u * gamma),
                (x, y) if x < 2 && y < 2 => alone[(x, y)],
                _ => C64::ZERO,
            });
            let region = model.point_region([0.0; 3], 0.0);
            let gate = model.folds(b, 0, &region, usize::MAX);
            let exists = if atom { gate[0] < EXISTS && gate[1] < EXISTS } else { gate[2] < EXISTS && gate[3] < EXISTS };
            let mut errors = [0.0; 2];
            let mut booked = [0.0; 2];
            if !exists {
                return (exists, errors, booked);
            }
            for (i, (cluster, place, level)) in [(cg, 0, 0), (cb, 0, 2)].into_iter().enumerate() {
                let (lambda, psi) = eigenvalue(&model, cluster, place);
                let ground = psi[0].norm_sqr() / psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
                errors[i] = lambda.im / like(&exact, level).0.im - 1.0;
                let sample = Sample { cluster, at: Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, psi };
                let of = model.errors(&[unit.clone(), Unit::Second { ground: b }], &[sample]);
                booked[i] = (of[0][0].population + ORDER_FORCE * of[1][0].population) / ground;
            }
            (exists, errors, booked)
        };
        for atom in [false, true] {
            let mut refused = vec![];
            for beta in [10.0, 6.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5, 0.1, 0.02] {
                let (exists, errors, booked) = case(0.5, 1.0, 0.2, beta, atom);
                println!("atom {atom} β {beta:5.2}: exists {exists}; kept line's rate {:+.2e} (booked {:.2e}), b's rate {:+.2e} (booked {:.2e})", errors[0], booked[0], errors[1], booked[1]);
                if !exists {
                    refused.push(beta);
                    continue;
                }
                for i in 0..2 {
                    assert!(errors[i].abs() <= booked[i], "atom {atom}, β = {beta} Γ, rate {i}: {:+.3e} of the exact one, booked {:.3e}", errors[i], booked[i]);
                }
            }
            assert!([1.2, 1.0, 0.8].iter().all(|beta| refused.contains(beta)), "atom {atom}: around the Raman resonance the fold is refused: {refused:?}");
            let far: &[f64] = if atom { &[10.0, 6.0, 3.0, 2.0, 0.02] } else { &[10.0, 6.0, 3.0, 2.0] };
            assert!(far.iter().all(|beta| !refused.contains(beta)), "atom {atom}: far from the Raman resonance the fold exists: {refused:?}");
        }
        let (exists, errors, _) = case(2.0, 0.0, 0.2, 27.4, true);
        assert!(exists && errors[1].abs() < 1e-3, "b's rate through the dressed e: {:+.2e} of the exact one", errors[1]);
        let (_, errors, booked) = case(2.0, 0.0, 0.2, 27.4, false);
        assert!(errors[1].abs() <= booked[1] && errors[1] < -5e-3, "with the bare levels {:+.2e}, booked {:.2e}", errors[1], booked[1]);
    }

    /// What a folded drive does to its ground level is within its own gate's bounds (module documentation, # What a
    /// folded drive does to its level): on the lin-perp-lin scene at 160 mW/cm² a beam, closed, at points across a fringe
    /// of the cooling light and off the trap's axis, the shift of every ground state by its drives folded at the atom --
    /// the elimination's own, −Re Σ_e V[e, g]* R[e, g] over the carriers -- is within `near_shift` on the point, and its
    /// slope along the fringes and across them (a central difference) within `near_slope`. The bounds are the drives'
    /// own: the shift's is below a twentieth of what the coupling would be across Γ/2, a resonant drive's distance.
    #[test]
    fn a_folded_drive_s_shift_and_its_slope_are_within_their_bounds() {
        let (model, centre) = closed("R4_LIN_4");
        let shift = |cluster: usize, place: usize, r: [f64; 3]| -> f64 {
            let c = &model.clusters[cluster];
            let psi: Vec<C64> = (0..c.states.len()).map(|x| if x == place { C64::ONE } else { C64::ZERO }).collect();
            let sample = Sample { cluster, at: Point { r, kappa: [0.0; 3], t: 0.0 }, psi };
            let at = model.measured(&sample, None);
            let g = c.states[place];
            let mut out = 0.0;
            for (cc, response) in model.carriers.iter().zip(&at.responses) {
                let Some(response) = response else { continue };
                for e in 0..EXCITED {
                    let v: C64 = cc.beams.iter().zip(&cc.folded).map(|(&l, m)| at.amplitudes[l].value * m[(GROUND + e, g)]).sum();
                    out -= (v.conj() * response[(e, place)]).re;
                }
            }
            out
        };
        let (mut largest, mut steepest, mut tightest) = (0.0_f64, 0.0_f64, f64::INFINITY);
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, 0.1], [0.0, 0.0, 0.17], [0.0, 0.0, 0.25], [0.3, 0.0, 0.07], [0.0, 0.4, 0.31]] {
            let r: [f64; 3] = std::array::from_fn(|i| centre[i] + offset[i]);
            let region = model.point_region(r, 0.0);
            for (cluster, c) in model.clusters.iter().enumerate() {
                for (place, &g) in c.states[..c.ground].iter().enumerate() {
                    let (here, bound) = (shift(cluster, place, r), model.near_shift(g, &region));
                    assert!(here.abs() <= bound * (1.0 + 1e-9), "the shift {here} of ground state {g} at {r:?} beyond its bound {bound}");
                    let coupling: f64 = model.across(g, &region).iter().zip(&region.forms.drives(&model)[g]).filter(|of| of.1.level == Level::Atom).map(|(of, d)| of.size * of.size / (0.5 * d.gamma)).sum();
                    assert!(bound <= coupling / 20.0, "the bound {bound} of ground state {g} against its coupling across Γ/2, {coupling}");
                    largest = largest.max(here.abs());
                    if here.abs() > 1.0 {
                        tightest = tightest.min(bound / here.abs());
                    }
                    let step = 1e-4;
                    for axis in [0, 2] {
                        let moved = |sign: f64| {
                            let mut at = r;
                            at[axis] += sign * step;
                            shift(cluster, place, at)
                        };
                        let slope = (moved(1.0) - moved(-1.0)) / (2.0 * step);
                        let bound = model.near_slope(g, &region);
                        assert!(slope.abs() <= bound * (1.0 + 1e-6) + 1e-6, "the slope {slope} of ground state {g}'s shift along axis {axis} at {r:?} beyond its bound {bound}");
                        steepest = steepest.max(slope.abs());
                    }
                }
            }
        }
        eprintln!("the largest shift {largest:.2} rad/µs, the steepest slope {steepest:.1} rad/µs/µm, the least bound over its shift {tightest:.2}");
        assert!(largest > 20.0 && steepest > 100.0, "the points read a light shift: {largest}, {steepest}");
    }

    /// The gate of a drive folded at the atom says that its first-order Doppler series converges (module documentation,
    /// # Levels). R2_RECOIL's σ⁺ probe, its drives folded at the atom, made weak and near its line: its beams' couplings
    /// scaled to |V| = Γ/10 on the atoms' |2,2⟩ → |3′,3′⟩, that excited level moved to put the drive Γ from its line at
    /// the trap's centre (D = |Γ − iΓ/2|). |V| below Γ/4: its share across the moving atom's actual distance -- never
    /// below Γ/2 -- is below ½ at any speed, and so is the tail it is the one share of: the gate before. At the speed
    /// where the photon's Doppler shift is 1.5 D (the series in w/D diverges) the gate, the larger of the tail and the
    /// series' one gate of |w| across D, is beyond ½; at rest it is the tail.
    #[test]
    fn a_drive_at_the_atom_fails_where_its_doppler_series_diverges() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R2_RECOIL.json");
        let load = || Request::from_native(&serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()).unwrap();
        let at_the_atom = |request: &Request| -> Model {
            let light = Light::new(request);
            let bare = Model::new(request, &light, &[]).unwrap();
            let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
            Model::new(request, &light, &kept).unwrap()
        };
        // The strongest drive at the atom of the ground state g at the trap's centre: the centre, its place among g's
        // drives, its component, excited state and width, its size and its detuning there.
        let request = load();
        let g = (0..GROUND).find(|&s| request.initial_density[(s, s)].re == 1.0).unwrap();
        let strongest = |model: &Model| -> ([f64; 3], usize, (usize, usize, f64), Across) {
            let centre = model.trap(&std::array::from_fn(|s| if s == g { 1.0 } else { 0.0 }), &model.prepared()).unwrap().centre;
            let region = model.point_region(centre, 0.0);
            let drives = &region.forms.drives(model)[g];
            let place = (0..drives.len()).filter(|&i| drives[i].level == Level::Atom).max_by(|&a, &b| model.across(g, &region)[a].size.total_cmp(&model.across(g, &region)[b].size)).expect("a drive folded at the atom");
            (centre, place, (drives[place].component, drives[place].excited, drives[place].gamma), model.across(g, &region)[place])
        };
        let (_, _, (component, excited, gamma), of) = strongest(&at_the_atom(&request));
        let mut weak = load();
        let light = Light::new(&weak);
        for &l in &light.frequencies[component / 2].beams {
            let beam = &mut weak.beams[l];
            for z in beam.coupling.data.iter_mut().chain(beam.counter_coupling.data.iter_mut()) {
                *z = *z * (0.1 * gamma / of.size);
            }
        }
        weak.hamiltonian[(GROUND + excited, GROUND + excited)] += C64::real(gamma - of.centre);
        let model = at_the_atom(&weak);
        let (centre, place, (component, _, gamma), of) = strongest(&model);
        let group = model.point_region(centre, 0.0).forms.drives(&model)[g][place].group;
        let (size, distance, wave) = (of.size, of.distance_at_rest(gamma), model.wave_of(component));
        let speed = 1.5 * distance / wave;
        let (rest, moving) = (model.point_region(centre, 0.0), model.point_region(centre, speed));
        let w = wave * speed + 0.5 * model.hbar_over_mass * wave * wave;
        let [gate, _, _, _, tail] = model.folds(g, group, &moving, usize::MAX);
        let share = 2.0 * (size / model.across(g, &moving)[place].distance(gamma)).powi(2);
        eprintln!("|V| {:.3} Γ, D {:.3} Γ, k {wave:.3}/µm; at {speed:.2} µm/µs |w/D| {:.2}: its share {share:.3}, the tail {tail:.3}, its gate {gate:.3}", size / gamma, distance / gamma, w / distance);
        assert!(size < 0.25 * gamma && w / distance > 1.0, "|V| {size} against Γ {gamma}, |w/D| {}", w / distance);
        assert!(share < 0.5 && tail < 0.5, "before: its share {share}, the tail {tail}");
        assert!(gate >= 0.5, "its gate {gate} with the Doppler series diverging");
        let [still, _, _, _, still_tail] = model.folds(g, group, &rest, usize::MAX);
        eprintln!("at rest: its gate {still:.4}, the tail {still_tail:.4}");
        assert!(still < 0.5 && still_tail < 0.5 && still >= still_tail, "at rest: its gate {still}, the tail {still_tail}");
    }

    /// The slow states of a cluster's generator at a point, the atom at rest: its eigenvalues nearest the real axis, as
    /// many as it has ground states, by their real parts.
    fn slow(model: &Model, cluster: usize, r: [f64; 3]) -> Vec<C64> {
        use crate::step::Middle as _;
        let c = &model.clusters[cluster];
        let middle = model.assemble(cluster, r, 0.0, &model.running());
        let mut h = Mat::zeros(c.states.len(), c.states.len());
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
        let mut values = crate::linalg::eigen(&h).expect("a cluster's generator has its eigenvalues").values;
        values.sort_by(|a, b| b.im.total_cmp(&a.im));
        values.truncate(c.ground);
        values.sort_by(|a, b| a.re.total_cmp(&b.re));
        values
    }

    /// The exact elimination's slow states at a point, the atom at rest: of the no-jump generator of a cluster's ground
    /// states with every excited state -- the cluster's own generator without its drives folded at the atom, those
    /// drives' couplings V, and the excited block K − δ the eliminations take there --, the eigenvalues nearest the real
    /// axis, by their real parts.
    fn eliminated(model: &Model, cluster: usize, carrier: usize, r: [f64; 3]) -> Vec<C64> {
        let n = model.clusters[cluster].ground;
        let mut values = crate::linalg::eigen(&no_jump(model, cluster, carrier, r)).expect("the generator has its eigenvalues").values;
        values.sort_by(|a, b| b.im.total_cmp(&a.im));
        values.truncate(n);
        values.sort_by(|a, b| a.re.total_cmp(&b.re));
        values
    }

    /// The no-jump generator of a cluster's ground states with every excited state at a point, the atom at rest
    /// (`eliminated`): the ground states first.
    fn no_jump(model: &Model, cluster: usize, carrier: usize, r: [f64; 3]) -> Mat {
        use crate::step::Middle as _;
        let (c, cc) = (&model.clusters[cluster], &model.carriers[carrier]);
        let n = c.ground;
        assert!(c.frames.iter().all(|frame| *frame == c.frames[0]), "the cluster's ground states share a frame");
        let amplitudes = model.light.amplitudes(r);
        let middle = model.middle_of(&model.plan(cluster, &model.running()), 0.0, &amplitudes, None);
        let mut ground = Mat::zeros(n, n);
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut ground);
        let h = &model.far_hamiltonian(&amplitudes, &model.running(), 0, &model.coherent_dressing())[0];
        let frame = model.omega(&c.frames[0]) + cc.detuning;
        Mat::from_fn(n + EXCITED, n + EXCITED, |x, y| match (x < n, y < n) {
            (true, true) => ground[(x, y)],
            (false, true) => cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + x - n, c.states[y])]).sum(),
            (true, false) => cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + y - n, c.states[x])]).sum::<C64>().conj(),
            (false, false) => h[(GROUND + x - n, GROUND + y - n)] - model.decay[(x - n, y - n)] * C64::new(0.0, 0.5) - if x == y { C64::real(frame) } else { C64::ZERO },
        })
    }

    /// Where a jump lands, against the exact elimination (model.rs `Model::landed`; assemble.rs, # The fourth order): the
    /// lin-perp-lin scenes, the cluster F=2 m = −2, 0, 2 with its D2 drives at the atom at the fourth order. A bare
    /// ground state f is the slow subspace's part of it and a fast part that decays at once: exactly, the part is
    /// P (f, 0) with P the spectral projector of the no-jump generator onto its three slow eigenvectors, the prompt
    /// photon's probability 1 − ‖P (f, 0)‖²/‖f‖², and the slow state's coordinates (1 + ½X) times the part's ground
    /// amplitudes. The model's p and Π f are these to the next order, s² -- p itself is about half the landing state's
    /// share --, and the norm is kept on average by construction: ‖Π f‖² + p ‖f‖² = ‖f‖².
    #[test]
    fn a_jump_s_landing_is_the_exact_elimination_s_slow_part() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (closed, centre) = closed(scene);
            let states = closed.clusters[closed.membership[3]].states.clone();
            let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
            let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
            let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }, Unit::Second { ground: 3 }]).collect();
            let fourth = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
            let (cluster, carrier) = (fourth.membership[3], fourth.carrier_of(cooling.frequency).unwrap());
            let plan = fourth.plan(cluster, &fourth.running());
            assert!(plan.fourth.is_some(), "{scene}: the cluster's fold is the fourth order's");
            let (mut s, mut worst, mut turned) = (0.0_f64, 0.0_f64, 0.0_f64);
            for offset in [0.0, 0.05, 0.1, 0.17, 0.26] {
                let r = [centre[0], centre[1], centre[2] + offset];
                let at = Point { r, kappa: [0.0; 3], t: 0.0 };
                let exact = crate::linalg::eigen(&no_jump(&fourth, cluster, carrier, r)).unwrap();
                let mut order: Vec<usize> = (0..exact.values.len()).collect();
                order.sort_by(|&a, &b| exact.values[b].im.total_cmp(&exact.values[a].im));
                for f in [[C64::ONE, C64::ZERO, C64::ZERO], [C64::ZERO, C64::ONE, C64::ZERO], [C64::real(0.6), C64::new(0.0, 0.5), C64::real(-0.4)]] {
                    let size = crate::linalg::norm_sqr(&f);
                    // P (f, 0) = Σ_slow r_i (R⁻¹ (f, 0))_i.
                    let mut part = vec![C64::ZERO; 3 + EXCITED];
                    for &i in &order[..3] {
                        let weight: C64 = (0..3).map(|y| exact.inverse[(i, y)] * f[y]).sum();
                        for (x, z) in part.iter_mut().enumerate() {
                            *z += exact.vectors[(x, i)] * weight;
                        }
                    }
                    let prompt = 1.0 - crate::linalg::norm_sqr(&part) / size;
                    let slow = fourth.landed(&plan, None, &at, &f);
                    let p = 1.0 - crate::linalg::norm_sqr(&slow) / size;
                    worst = worst.max((p - prompt).abs());
                    s = s.max(2.0 * prompt);
                    // The slow state's direction: Π f against the exact part's ground amplitudes (to their common first
                    // order they differ by the metric's half, of the order of p itself; the angle between them is s²).
                    let overlap: C64 = (0..3).map(|x| slow[x].conj() * part[x]).sum();
                    let ground: f64 = part[..3].iter().map(|z| z.norm_sqr()).sum();
                    turned = turned.max(1.0 - overlap.norm_sqr() / (crate::linalg::norm_sqr(&slow) * ground));
                }
            }
            eprintln!("{scene}: the largest prompt photon's probability {:.4}; the model's differs by at most {worst:.2e}; the slow state turned by at most {turned:.2e}", 0.5 * s);
            assert!(worst <= 1.5 * s * s, "{scene}: p within the next order: {worst} for s = {s}");
            assert!(turned <= s * s, "{scene}: the slow state within the next order: {turned} for s = {s}");
        }
    }

    /// The fold's orders against the exact elimination, the atom frozen (module documentation, # The fold's order;
    /// assemble.rs, # The fourth order): the lin-perp-lin scenes at 40, 80 and 160 mW/cm² a beam, 20 linewidths red, on
    /// ⁸⁷Rb's real F′ = 1, 2, 3 in the tweezer. The cluster F=2 m = −2, 0, 2 with every cooling drive into F′=3 folded at
    /// the atom, at the second order and at the fourth, against the slow eigenvalues of the same ground states with all
    /// the excited states (`eliminated`; the dressing coherent, so that the levels are the same in both), at points across
    /// a fringe. Of the three slow states, with s the largest share 2 Σ_e |R[e, g]|² met: the potentials err by s/2 of the
    /// light's largest shift there at the second order and by s²/2 at the fourth; the force on them (the potentials'
    /// slope along the fringes, a central difference), relative to the steepest potential met, by s and by 3/2 s²; their
    /// decay rates by 3/2 s and by the next order. And the fourth order's generator is a Lindblad one as the second's: a
    /// state's norm falls at its photons' rate.
    #[test]
    fn the_fourth_order_s_potentials_forces_and_rates_against_the_exact_elimination() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (closed, centre) = closed(scene);
            let states = closed.clusters[closed.membership[3]].states.clone();
            assert_eq!(states, [3, 5, 7], "the cluster of F=2 m = −2, 0, 2");
            // Every cooling drive from the cluster into D2 -- F′ = 1, 2, 3 -- at the atom (the closure leaves the weaker
            // bare), the dressing coherent.
            let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
            let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
            let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }]).collect();
            let second = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
            let fourth = second.rekept(&[second.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
            let (cluster, carrier) = (second.membership[3], second.carrier_of(cooling.frequency).unwrap());
            assert_eq!(second.fold_frequency(cluster), Some(cooling.frequency));
            assert_eq!((second.fourth[cluster], fourth.fourth[fourth.membership[3]]), (None, Some(cooling.frequency)));
            assert!(second.units().contains(&Unit::Second { ground: 3 }) && !fourth.units().contains(&Unit::Second { ground: 3 }));
            let (mut s, mut largest, mut steepest) = (0.0_f64, 0.0_f64, 0.0_f64);
            let (mut potential, mut force, mut rate) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
            let step = 1e-4;
            for offset in [0.0, 0.03, 0.07, 0.1, 0.15, 0.2, 0.26, 0.33] {
                let r = [centre[0], centre[1], centre[2] + offset];
                let read = |values: &dyn Fn(f64) -> Vec<C64>| -> (Vec<C64>, Vec<f64>) {
                    let (up, down) = (values(step), values(-step));
                    (values(0.0), up.iter().zip(&down).map(|(a, b)| (a.re - b.re) / (2.0 * step)).collect())
                };
                let (exact, pull) = read(&|by| eliminated(&second, cluster, carrier, [r[0], r[1], r[2] + by]));
                let level = {
                    use crate::step::Middle as _;
                    let middle = second.middle_of(&second.plan(cluster, &second.running()), 0.0, &second.light.amplitudes(r), None);
                    let mut h = Mat::zeros(3, 3);
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                    crate::linalg::eigen(&h).unwrap().values
                };
                let mean = level.iter().map(|z| z.re).sum::<f64>() / 3.0;
                let shift = exact.iter().map(|z| (z.re - mean).abs()).fold(0.0, f64::max);
                let decay = exact.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
                largest = largest.max(shift);
                steepest = pull.iter().map(|f| f.abs()).fold(steepest, f64::max);
                for (order, model) in [&second, &fourth].into_iter().enumerate() {
                    let (values, slopes) = read(&|by| slow(model, model.membership[3], [r[0], r[1], r[2] + by]));
                    for i in 0..3 {
                        potential[order] = potential[order].max((values[i].re - exact[i].re).abs() / shift);
                        force[order] = force[order].max((slopes[i] - pull[i]).abs());
                        rate[order] = rate[order].max((values[i].im - exact[i].im).abs() / decay);
                    }
                }
                // The largest share at the point, and the fourth order's hazard identity on a state of the cluster.
                let psi: Vec<C64> = (0..3).map(|x| C64::cis(1.3 * x as f64) * (0.2 + 0.3 * x as f64)).collect();
                let here = Point { r, kappa: [0.0; 3], t: 0.0 };
                let sample = Sample { cluster, at: here, psi: psi.clone() };
                let at = second.measured(&sample, None);
                let response = at.responses[carrier].as_ref().unwrap();
                s = (0..3).map(|y| 2.0 * (0..EXCITED).map(|x| response[(x, y)].norm_sqr()).sum::<f64>()).fold(s, f64::max);
                let plan = fourth.plan(fourth.membership[3], &fourth.running());
                let photons: f64 = fourth.photons_of(&plan, None, &here, &psi, crate::model::photon::Gradients::None).0.iter().map(|p| p.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum::<f64>()).sum();
                let mut h = Mat::zeros(3, 3);
                {
                    use crate::step::Middle as _;
                    let middle = fourth.assemble(fourth.membership[3], r, 0.0, &fourth.running());
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                }
                let falls: f64 = (0..3).flat_map(|x| (0..3).map(move |y| (x, y))).map(|(x, y)| (psi[x].conj() * C64::I * (h[(x, y)] - h[(y, x)].conj()) * psi[y]).re).sum();
                // The potential is the same generator's (V_L is the light set's less the same without the trap light): a
                // state's potential differs between the orders by the diagonals of what the fourth adds to the two
                // generators, and its gradient is its own slope.
                let dark = crate::model::Lights((0..fourth.light.frequencies.len()).map(|w| if w == fourth.light.trap { crate::model::Lit::Off } else { crate::model::Lit::Coherent }).collect());
                let diagonal = |model: &Model, lights: &crate::model::Lights, place: usize| -> f64 {
                    use crate::step::Middle as _;
                    let middle = model.assemble(model.membership[3], r, 0.0, lights);
                    let mut g = Mat::zeros(3, 3);
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut g);
                    g[(place, place)].re
                };
                for (place, &state) in states.iter().enumerate() {
                    let alone: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
                    let of = |model: &Model, by: f64| model.potential([r[0], r[1], r[2] + by], &alone, &model.running());
                    let lit = fourth.running();
                    let more = diagonal(&fourth, &lit, place) - diagonal(&second, &lit, place) - (diagonal(&fourth, &dark, place) - diagonal(&second, &dark, place));
                    let added = of(&fourth, 0.0) - of(&second, 0.0);
                    assert!((added - more).abs() <= 1e-9 * largest.max(1.0), "{scene}: the potential's fourth order {added} against the generator's {more}");
                    let slope = fourth.expand(r, &alone, &fourth.running(), 1, None).quadratic.gradient[2];
                    let difference = (of(&fourth, step) - of(&fourth, -step)) / (2.0 * step);
                    // (The central difference's own error: (2k step)²/6 of the slope, 4e-7.)
                    assert!((slope - difference).abs() <= 2e-6 * slope.abs() + 1e-6, "{scene}: the potential's gradient {slope} against its slope {difference}");
                }
                // To the product of the fourth order and the first-order Doppler term (here the recoil shift, 3e-5 of the
                // detuning, times s), which the photons' product holds and the middle's decay, made at rest, does not.
                assert!((falls - photons).abs() <= 1e-5 * photons, "{scene}: the norm falls at {falls}, the photons' rate is {photons}");
            }
            let force = force.map(|f| f / steepest);
            eprintln!("{scene}: s = {s:.4}, the largest shift {largest:.2} rad/µs: potentials {:.2e} (second), {:.2e} (fourth); forces {:.2e}, {:.2e}; decay rates {:.2e}, {:.2e}", potential[0], potential[1], force[0], force[1], rate[0], rate[1]);
            assert!((0.4 * s..0.6 * s).contains(&potential[0]) && potential[1] <= 0.6 * s * s, "{scene}: the potentials err by s/2 and by s²/2: {potential:?} for s = {s}");
            assert!((0.5 * s..1.0 * s).contains(&force[0]) && force[1] <= 1.5 * s * s, "{scene}: the forces err by s and by 3/2 s² at most: {force:?} for s = {s}");
            assert!((1.2 * s..1.8 * s).contains(&rate[0]) && rate[1] <= 2.5 * s * s, "{scene}: the decay rates err by 3/2 s and by the next order: {rate:?} for s = {s}");
        }
    }

    /// A lin-perp-lin scene's cycling point: where the cooling light at the trap's axis is σ⁻ alone, F=2 m=−2 is driven
    /// into F′=3 m′=−3 and nowhere else, and that state decays to it alone -- a two-level atom. The scene's model with
    /// every cooling drive of m=−2's cluster at the atom and the dressing kept, at the second order and at the fourth;
    /// the point; and of the no-jump generator there the drive's share s = 2|V|²/(Δ² + Γ²/4), the two-level atom's
    /// steady photon rate Γ |V|²/(Δ² + Γ²/4 + 2|V|²) with the ground state's own bare scattering, and the line's
    /// (Γ/2Δ)².
    fn cycling(scene: &str) -> (Model, Model, Point, [f64; 3]) {
        let (closed, centre) = closed(scene);
        let states = closed.clusters[closed.membership[3]].states.clone();
        let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
        let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
        let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }]).collect();
        let second = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
        let fourth = second.rekept(&[second.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
        let (cluster, carrier) = (second.membership[3], second.carrier_of(cooling.frequency).unwrap());
        assert_eq!(second.clusters[cluster].states[0], 3, "the cluster's first place is F=2 m=−2");
        // The point where the light is σ⁻ alone: where m=−2's coupling is the largest (its σ⁺ part is nothing there).
        // The excited state it drives (in the request's basis a combination of states), its detuning and its width.
        let column = |z: f64| -> (f64, Mat) {
            let g = no_jump(&second, cluster, carrier, [centre[0], centre[1], z]);
            ((0..EXCITED).map(|e| g[(3 + e, 0)].norm_sqr()).sum(), g)
        };
        let grid: Vec<f64> = (0..400).map(|i| centre[2] + 0.4 * i as f64 / 400.0).collect();
        let best = grid.iter().copied().max_by(|a, b| column(*a).0.total_cmp(&column(*b).0)).unwrap();
        let (mut lo, mut hi) = (best - 0.001, best + 0.001);
        for _ in 0..60 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if column(a).0 > column(b).0 { hi = b } else { lo = a }
        }
        let z = 0.5 * (lo + hi);
        let (v2, g) = column(z);
        let bright: Vec<C64> = (0..EXCITED).map(|e| g[(3 + e, 0)] * (1.0 / v2.sqrt())).collect();
        let level: C64 = (0..EXCITED * EXCITED).map(|i| bright[i / EXCITED].conj() * g[(3 + i / EXCITED, 3 + i % EXCITED)] * bright[i % EXCITED]).sum();
        let (gamma, delta) = (-2.0 * level.im, level.re - g[(0, 0)].re);
        let s = 2.0 * v2 / (delta * delta + 0.25 * gamma * gamma);
        let exact = gamma * v2 / (delta * delta + 0.25 * gamma * gamma + 2.0 * v2) - 2.0 * g[(0, 0)].im;
        let at = Point { r: [centre[0], centre[1], z], kappa: [0.0; 3], t: 0.0 };
        (second, fourth, at, [s, exact, 0.25 * gamma * gamma / (delta * delta)])
    }

    /// A cycling state's photons per unit time, against the two-level atom's (module documentation, # The fold's order;
    /// `cycling`). With the drive's share s: the second-order fold's rate is too high by s; the fourth order's jumps
    /// alone are too low by s/2; with the landing -- every jump back into the bare state followed by one more photon
    /// with the probability p, and so on: 1/(1 − p) photons a jump, these lines being far from resonance -- the rate is
    /// right to the next order: low by s², to the order after that, within `ORDER_RATES` s².
    #[test]
    fn a_cycling_state_s_photon_rate_with_the_landing_is_the_two_level_atom_s() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (second, fourth, at, [s, exact, width]) = cycling(scene);
            let psi = [C64::ONE, C64::ZERO, C64::ZERO];
            let rate = |model: &Model| -> f64 {
                let plan = model.plan(model.membership[3], &model.running());
                model.photons_of(&plan, None, &at, &psi, crate::model::photon::Gradients::None).0.iter().map(|p| p.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum::<f64>()).sum()
            };
            let plan = fourth.plan(fourth.membership[3], &fourth.running());
            let prompt = 1.0 - crate::linalg::norm_sqr(&fourth.landed(&plan, None, &at, &psi));
            assert!(prompt > 0.0, "{scene}: far from its line the landing's norm is below one: p = {prompt}");
            let errors = [rate(&second) / exact - 1.0, rate(&fourth) / exact - 1.0, rate(&fourth) / (1.0 - prompt) / exact - 1.0];
            // What the comparison cannot resolve: the second order's rate is the two-level atom's times (1 + s) but for
            // the line's width over its detuning, squared.
            let floor = (errors[0] - s).abs();
            let rest = (errors[2] - (errors[0] - s)) / (s * s);
            eprintln!("{scene}: s = {s:.4}, p = {prompt:.4}: the photon rate against the two-level atom's -- second order {:+.3e} (s {:+.1e}), fourth {:+.3e}, fourth with the landing {:+.3e} ({rest:+.3} s²)", errors[0], errors[0] - s, errors[1], errors[2]);
            assert!(floor <= width, "{scene}: the second order's rate is too high by s: {} for s = {s}", errors[0]);
            assert!((errors[1] + 0.5 * s).abs() <= ORDER_RATES * s * s + floor, "{scene}: the fourth order's jumps alone too low by s/2: {} for s = {s}", errors[1]);
            assert!((rest + 1.0).abs() <= s && rest.abs() <= ORDER_RATES, "{scene}: with the landing the rate is low by s², to the order after: {} for s = {s}", errors[2]);
        }
    }

    /// The landing's norm against the threshold (model.rs `Model::landed`, step.rs `jump`) is the two-level atom's rate
    /// to the fourth order on and off resonance, by one rule. Two exact levels (`three_levels` with no second light:
    /// g –V→ e alone, the drive folded at the atom at the fourth order, the atom at κ = −k/2 where its Doppler shift is
    /// nothing), Δ from 0 to 6 Γ on both sides, s = 2|V|²/(Δ² + Γ²/4) = 0.03, 0.06, 0.12. With the slow state's rate γ off
    /// the generator and the landing's norm n = ‖Π g‖², the waiting time after a jump is P(T > t) = min(1, n e^{−γt}), its
    /// mean (max(0, ln n) + min(1, n))/γ, against the exact steady rate Γ|V|²/(Δ² + Γ²/4 + 2|V|²):
    /// - n = 1 − p with p = (s/2)(Δ² − 3Γ²/4)/(Δ² + Γ²/4) to its order: above one within 0.866 Γ of the line (1 + 3s/2 on
    ///   resonance), below one beyond;
    /// - the rate is the exact one within `ORDER_RATES` s² everywhere (+1.07 s² on resonance at s = 0.03, −0.99 s² at
    ///   6 Γ); with the norm clamped at one it was 1.52 s high on resonance;
    /// - `jump` itself on resonance: every jump emits one photon and no more, and a trajectory that lands back in g goes
    ///   on with the norm n, above one.
    #[test]
    fn the_landing_s_norm_against_the_threshold_is_the_two_level_atom_s_rate_on_and_off_resonance() {
        use crate::step::{self, Dynamics, Trajectory};
        let (g, e) = (THREE[0], THREE[2]);
        for s in [0.03, 0.06, 0.12] {
            for delta in [0.0, 0.25, -0.25, 0.5, 0.8, 1.0, -1.0, 2.0, -6.0] {
                let v = (0.5 * s * (delta * delta + 0.25_f64)).sqrt();
                let (request, of, gamma) = three_levels(v, delta, 0.0, 0.0);
                let light = Light::new(&request);
                let component = Component { frequency: of[0], counter: false };
                let model = Model::new(&request, &light, &[Unit::Bare { component, excited: e, ground: g }, Unit::Second { ground: g }]).unwrap();
                let cluster = model.membership[g];
                assert!(model.clusters[cluster].states == [g] && model.fourth[cluster].is_some(), "g alone, its fold at the fourth order");
                let beam = &light.beams[light.frequencies[of[0]].beams[0]];
                let k = beam.wave_vector();
                let at = Point { r: [0.0; 3], kappa: k.map(|x| -0.5 * x), t: 0.0 };
                let rate = {
                    use crate::step::Middle as _;
                    let middle = model.assemble(cluster, at.r, 0.0, &model.running());
                    let mut h = Mat::zeros(1, 1);
                    middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
                    -2.0 * h[(0, 0)].im
                };
                let plan = model.plan(cluster, &model.running());
                let n = crate::linalg::norm_sqr(&model.landed(&plan, None, &at, &[C64::ONE]));
                let p = 0.5 * s * (delta * delta - 0.75) / (delta * delta + 0.25);
                assert!((1.0 - n - p).abs() <= 1.5 * s * s, "s = {s}, Δ = {delta} Γ: the landing's norm {n} against 1 − p, p = {p}");
                assert!((n > 1.0) == (delta.abs() < 0.75_f64.sqrt()), "s = {s}, Δ = {delta} Γ: the norm {n} is above one within 0.866 Γ of the line");
                let (vr, d) = (v * gamma, delta * gamma);
                let exact = gamma * vr * vr / (d * d + 0.25 * gamma * gamma + 2.0 * vr * vr);
                let mean = (n.ln().max(0.0) + n.min(1.0)) / rate;
                let error = 1.0 / (mean * exact) - 1.0;
                println!("s = {s}, Δ = {delta:+.2} Γ: n − 1 = {:+.3} s, the rate {:+.3} s² of the exact one (the norm clamped at one: {:+.3} s)", (n - 1.0) / s, error / (s * s), (rate / n.min(1.0) / exact - 1.0) / s);
                assert!(error.abs() <= ORDER_RATES * s * s, "s = {s}, Δ = {delta} Γ: the rate {error:+.3e} of the exact one, s² = {:.3e}", s * s);
                if delta == 0.0 {
                    assert!(((rate / exact - 1.0) / s - 1.5).abs() < 0.1, "on resonance the clamped norm's rate is 1.5 s high");
                    let mut reference = [0.0; GROUND];
                    reference[g] = 1.0;
                    let run = crate::model::Run::new(&model, reference);
                    let mut back = 0;
                    for index in 0..200 {
                        let mut t = Trajectory::new(&run, g, at.r, at.kappa, 11, index);
                        step::jump(&run, &mut t).unwrap();
                        assert_eq!((t.work / run.photon_work()).round(), 1.0, "one photon a jump where the landing's norm is above one");
                        let norm = crate::linalg::norm_sqr(&t.psi);
                        if t.cluster == cluster {
                            back += 1;
                            assert!(norm > 1.0 && (norm - n).abs() <= 0.2 * (n - 1.0), "the trajectory goes on with the landing's norm: {norm} against {n}");
                        } else {
                            assert!((norm - 1.0).abs() <= 1e-12, "a second-order landing is the bare state: {norm}");
                        }
                    }
                    assert!(back > 0, "some jumps land back in g");
                }
            }
        }
    }

    /// One ground state driven by one frequency to several excited states (`three_levels`' atom, no second light): g
    /// –V_i→ e_i, each line (its coupling, its detuning, in Γ) one drive folded at the atom at the fourth order. The
    /// request; the frequency; the excited states with their own widths (the scene's, equal to its eight digits); Γ.
    fn fan(lines: &[(f64, f64)]) -> (Request, usize, Vec<(usize, f64)>, f64) {
        let (mut request, of, gamma) = three_levels(0.0, 0.0, 0.0, 0.0);
        let light = Light::new(&request);
        let nu = Model::new(&request, &light, &[]).unwrap().tones[of[0]].nu[0];
        let width = |e: usize| -> f64 { request.emission.iter().map(|c| (0..GROUND).map(|s| c[(s, e)].norm_sqr()).sum::<f64>()).sum() };
        let excited: Vec<(usize, f64)> = (0..lines.len()).map(|i| (GROUND + i, width(GROUND + i))).collect();
        for (&(e, width), &(v, delta)) in excited.iter().zip(lines) {
            assert!((width - gamma).abs() <= 1e-6 * gamma, "the excited states share one width: {width} against {gamma}");
            request.beams[light.frequencies[of[0]].beams[0]].coupling[(e, THREE[0])] = C64::real(v * gamma);
            request.hamiltonian[(e, e)] = C64::real(delta * gamma + nu);
        }
        (request, of[0], excited, gamma)
    }

    /// What the fourth order leaves of a ground state's several drives (module documentation, # The fold's order), against
    /// the exact atom: g driven by one frequency to one, two and three excited states (`fan`; each line its share s =
    /// 2|V|²/(Δ² + Γ²/4) and its detuning in Γ), every drive folded at the atom at the fourth order, the atom where its
    /// Doppler shift is nothing. The rate as the two-level test above reads it -- the mean waiting time after a jump
    /// with the slow state's rate and the landing's norm against the threshold -- against the exact mean waiting time
    /// from g, ∫ ‖e^{−iHt} g‖² dt of the no-jump generator of g with its excited states; the potential, the generator's
    /// real part against the exact slow eigenvalue's, over the light's shift Σ |V|²/|Δ − iΓ/2|, of which the force's
    /// error is three times (a sixth order in the field against the second).
    /// - At one detuning the drives are one line of the share Σs: the rate's remainder is +0.87 to +1.07 (Σs)² on
    ///   resonance and −0.98 to −0.99 (Σs)² at 6 Γ whatever the number of drives and their shares, within `ORDER_RATES`
    ///   (Σs)² -- and 1.7 to 2.6 times `ORDER_RATES` Σs² (the drives booked one by one) with two and three of them.
    /// - At different distances (Σs)² is no bound: shares of 0.04 at 3 Γ and at 30 Γ leave −7.2 (Σs)², the near line
    ///   reading the far one's light shift over its own distance (L = 5.4 Σs).
    /// - Everywhere the remainder is within what the drives' units book (`Model::drive_error`, p_g = 1): the rate's
    ///   within the populations' (1.05 to 4.3 of it where every parameter is small; 26 of it where the far line moves
    ///   the near one by 1.8 of its distance, no small parameter: 0.61 booked), three times the potential's within T's.
    /// - The same atom with its fold at the second order (`Unit::Second` not kept: its jumps' rate Σ Γ |R|² against the
    ///   same waiting time, twice the potential's error for the force's): at one detuning the rate is high by Σs to the
    ///   widths' eight digits, and booked so; at different distances by more -- 2.72 Σs with shares of 0.04 at 3 Γ and
    ///   at 30 Γ, 3.10 Σs with 0.01 at 1 Γ and 0.05 at 40 Γ, 1.19 Σs at 20, 64 and 90 Γ --, within what the cluster's
    ///   unit books (`Model::order_error`: 3.21, 5.83, 1.26 Σs); the force's stays within Σs (0.94 Σs at most).
    #[test]
    fn a_ground_state_s_drives_leave_the_fourth_order_one_line_s_remainder() {
        let g = THREE[0];
        let together: Vec<Vec<(f64, f64)>> = [0.0, 0.5, -1.0, -6.0]
            .into_iter()
            .flat_map(|delta| [vec![(0.03, delta)], vec![(0.03, delta); 2], vec![(0.03, delta); 3], vec![(0.06, delta); 3], vec![(0.06, delta), (0.02, delta), (0.01, delta)]])
            .collect();
        let decade = vec![(0.04, -3.0), (0.04, -30.0)];
        let apart = [
            vec![(0.04, -3.0), (0.04, -3.5)],
            vec![(0.04, -3.0), (0.04, -6.0)],
            decade.clone(),
            vec![(0.04, -3.0), (0.04, 5.0)],
            vec![(0.04, 0.0), (0.04, -3.0)],
            vec![(0.01, -1.0), (0.05, -40.0)],
            vec![(0.06, -20.0), (0.006, -64.0), (0.002, -90.0)],
            vec![(0.03, -0.3), (0.03, 0.0), (0.03, 0.3)],
        ];
        for (index, case) in together.iter().chain(&apart).enumerate() {
            let lines: Vec<(f64, f64)> = case.iter().map(|&(s, delta)| ((0.5 * s * (delta * delta + 0.25_f64)).sqrt(), delta)).collect();
            let (request, of, excited, gamma) = fan(&lines);
            let light = Light::new(&request);
            let component = Component { frequency: of, counter: false };
            let kept: Vec<Unit> = excited.iter().map(|&(e, _)| Unit::Bare { component, excited: e, ground: g }).chain([Unit::Second { ground: g }]).collect();
            let model = Model::new(&request, &light, &kept).unwrap();
            let cluster = model.membership[g];
            assert!(model.clusters[cluster].states == [g] && model.fourth[cluster].is_some(), "g alone, its fold at the fourth order");
            let k = light.beams[light.frequencies[of].beams[0]].wave_vector();
            let at = Point { r: [0.0; 3], kappa: k.map(|x| -0.5 * x), t: 0.0 };
            let h = {
                use crate::step::Middle as _;
                let middle = model.assemble(cluster, at.r, 0.0, &model.running());
                let mut h = Mat::zeros(1, 1);
                middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
                h[(0, 0)]
            };
            let plan = model.plan(cluster, &model.running());
            let n = crate::linalg::norm_sqr(&model.landed(&plan, None, &at, &[C64::ONE]));
            let mean = (n.ln().max(0.0) + n.min(1.0)) / (-2.0 * h.im);
            // The exact no-jump generator of g and its excited states; its mean waiting time from g, M[g, g] of
            // i(H†M − MH) = −1 (M = ∫ e^{iH†t} e^{−iHt} dt); its slow eigenvalue.
            let m = lines.len() + 1;
            let exact = Mat::from_fn(m, m, |x, y| match (x, y) {
                (0, 0) => C64::ZERO,
                (0, y) => C64::real(lines[y - 1].0 * gamma),
                (x, 0) => C64::real(lines[x - 1].0 * gamma),
                (x, y) if x == y => C64::new(lines[x - 1].1 * gamma, -0.5 * excited[x - 1].1),
                _ => C64::ZERO,
            });
            let lyapunov = Mat::from_fn(m * m, m * m, |row, column| {
                let ((x, y), (a, b)) = ((row / m, row % m), (column / m, column % m));
                let mut z = C64::ZERO;
                if b == y {
                    z += C64::I * exact[(a, x)].conj();
                }
                if a == x {
                    z -= C64::I * exact[(b, y)];
                }
                z
            });
            let mut waiting: Vec<C64> = (0..m * m).map(|i| if i / m == i % m { C64::real(-1.0) } else { C64::ZERO }).collect();
            crate::linalg::Lu::new(&lyapunov).unwrap().solve(&mut waiting);
            let slow = crate::linalg::eigen(&exact).unwrap().values.into_iter().max_by(|a, b| a.im.total_cmp(&b.im)).unwrap();
            let rate = waiting[0].re / mean - 1.0;
            let shift: f64 = lines.iter().map(|&(v, delta)| v * v * gamma / (delta * delta + 0.25_f64).sqrt()).sum();
            let potential = (h.re - slow.re).abs() / shift;
            let (sum, squares): (f64, f64) = (case.iter().map(|c| c.0).sum(), case.iter().map(|c| c.0 * c.0).sum());
            let units: Vec<Unit> = excited.iter().map(|&(e, _)| Unit::Drive { component, excited: e, ground: g }).collect();
            let sample = Sample { cluster, at, psi: vec![C64::ONE] };
            let errors = model.errors(&units, &[sample]);
            let booked: [f64; 2] = [errors.iter().map(|e| e[0].population).sum(), errors.iter().map(|e| e[0].temperature).sum()];
            for (error, &(share, _)) in errors.iter().zip(case) {
                assert!((error[0].parameter - share).abs() <= 1e-6 * share, "{case:?}: a drive's share at the atom {} against {share}", error[0].parameter);
            }
            println!(
                "{case:?}: the rate {rate:+.3e} = {:+.3} (Σs)² = {:+.3} Σs², booked {:.3e}; three times the potential {:.3e} = {:.3} (Σs)², booked on T {:.3e}",
                rate / (sum * sum),
                rate / squares,
                booked[0],
                3.0 * potential,
                3.0 * potential / (sum * sum),
                booked[1]
            );
            assert!(rate.abs() <= booked[0], "{case:?}: the rate's remainder {rate:+.3e} beyond the drives' booked {:.3e}", booked[0]);
            assert!(3.0 * potential <= booked[1], "{case:?}: three times the potential's remainder {:.3e} beyond the drives' booked {:.3e} on T", 3.0 * potential, booked[1]);
            if index < together.len() {
                assert!(rate.abs() <= ORDER_RATES * sum * sum, "{case:?}: at one detuning the remainder is one line's of the share Σs: {:+.3} (Σs)²", rate / (sum * sum));
                let ends = case[0].1 == 0.0 || case[0].1 == -6.0;
                assert!(case.len() == 1 || !ends || rate.abs() > 1.5 * ORDER_RATES * squares, "{case:?}: and not the drives' one by one: {:+.3} Σs²", rate / squares);
            }
            if *case == decade {
                assert!(rate.abs() > 5.0 * ORDER_RATES * sum * sum, "{case:?}: a decade apart (Σs)² is no bound: {:+.3} (Σs)²", rate / (sum * sum));
            }
            // The second order's own (no `Unit::Second` kept): its jumps' rate against the exact waiting time, twice the
            // potential's error, and what the cluster's unit books of them.
            let second = Model::new(&request, &light, &kept[..kept.len() - 1]).unwrap();
            let lower = {
                use crate::step::Middle as _;
                let middle = second.assemble(second.membership[g], at.r, 0.0, &second.running());
                let mut h = Mat::zeros(1, 1);
                middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
                h[(0, 0)]
            };
            let state = Sample { cluster: second.membership[g], at, psi: vec![C64::ONE] };
            let order = second.errors(&[Unit::Second { ground: g }], &[state])[0][0];
            let (rate, force) = (waiting[0].re * (-2.0 * lower.im) - 1.0, 2.0 * (lower.re - slow.re).abs() / shift);
            println!("    its second order: the rate {rate:+.3e} = {:+.3} Σs, booked {:.3e}; twice the potential {force:.3e} = {:.3} Σs, booked on T {:.3e}", rate / sum, order.population, force / sum, order.temperature);
            // (The excited states' widths are one to the scene's eight digits: so are the one line's statements.)
            let exactly = 1.0 + 1e-6;
            assert!(rate.abs() <= order.population * exactly, "{case:?}: the second order's rate {rate:+.3e} beyond its booked {:.3e}", order.population);
            assert!(force <= order.temperature * exactly, "{case:?}: the second order's force {force:.3e} beyond its booked {:.3e} on T", order.temperature);
            if index < together.len() {
                assert!((rate - sum).abs() <= 1e-6 * sum && (order.population - sum).abs() <= 1e-6 * sum, "{case:?}: at one detuning the second order is high by Σs exactly, and booked so: {rate:e}, {:e}", order.population);
            }
            if *case == decade {
                assert!(rate > 2.0 * sum, "{case:?}: a decade apart Σs is no bound of the second order: {:+.3} Σs", rate / sum);
            }
        }
    }

    /// A jump at the fourth order (step.rs `jump`, the landing), on the cycling state at rest (`cycling`): of single jumps
    /// with their own streams, the photons a jump -- each one's counted work -- are 1/(1 − p) on average, within four of
    /// their mean's deviations (a geometric number's, √p/(1 − p) a jump).
    #[test]
    fn a_jump_at_the_fourth_order_draws_its_landing_s_photons() {
        use crate::step::{self, Dynamics, Trajectory};
        for scene in ["R4_LIN_1", "R4_LIN_4"] {
            let (_, fourth, at, _) = cycling(scene);
            let mut reference = [0.0; GROUND];
            reference[3] = 1.0;
            let run = crate::model::Run::new(&fourth, reference);
            let prompt = 1.0 - crate::linalg::norm_sqr(&run.landed(fourth.membership[3], &at, &[C64::ONE, C64::ZERO, C64::ZERO]));
            let trials = 4000u64;
            let mut photons = 0.0;
            for index in 0..trials {
                let mut t = Trajectory::new(&run, 3, at.r, [0.0; 3], 7, index);
                step::jump(&run, &mut t).unwrap();
                photons += (t.work / run.photon_work()).round();
            }
            let (mean, expected) = (photons / trials as f64, 1.0 / (1.0 - prompt));
            let deviation = prompt.sqrt() / (1.0 - prompt) / (trials as f64).sqrt();
            eprintln!("{scene}: p = {prompt:.4}: {mean:.4} photons a jump of {trials} jumps, 1/(1 − p) = {expected:.4} ± {deviation:.4}");
            assert!((mean - expected).abs() <= 4.0 * deviation, "{scene}: {mean} photons a jump against {expected} ± {deviation}");
        }
    }

    /// A middle made of given near forms carries the fourth order's part made of the same forms (assemble.rs
    /// `Model::middle_of`): on the lin-perp-lin scene with the cluster of F=2 m=−2 at the fourth order, at points along a
    /// fringe and off the axis, under the running lights and without the trap's, its generator on a state is the point's
    /// own middle's (the exact elimination's forms both) to their rounding, and differs from the second order's by the
    /// fourth order's size. So a table's error read against the exact elimination (tabulate.rs `Model::table_reads`) is
    /// the table's: of a grid with no axis -- the forms without the trap's light do not vary -- nothing, as it is at the
    /// second order; of one with axes, its interpolation's.
    #[test]
    fn a_middle_of_given_forms_carries_the_fourth_order() {
        let (closed, centre) = closed("R4_LIN_2");
        let second = closed.rekept(&closed.kept).unwrap();
        let fourth = closed.rekept(&[closed.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
        let cluster = fourth.membership[3];
        assert!(fourth.fourth[cluster].is_some() && second.fourth.iter().all(Option::is_none));
        let running = fourth.running();
        let mut untrapped = running.clone();
        untrapped.0[fourth.light.trap] = crate::model::Lit::Off;
        let n = fourth.clusters[cluster].states.len();
        let psi: Vec<C64> = (0..n).map(|x| C64::new(1.0 / (1.0 + x as f64), 0.3 * x as f64)).collect();
        let mut samples = vec![];
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.0, 0.07], [0.3, 0.0, 0.19], [0.0, 0.4, 0.31]] {
            let r: [f64; 3] = std::array::from_fn(|i| centre[i] + offset[i]);
            let sample = Sample { cluster, at: Point { r, kappa: [3.0, -2.0, 5.0], t: 0.0 }, psi: psi.clone() };
            for lights in [&running, &untrapped] {
                let amplitudes = fourth.light.amplitudes(r);
                let plan = fourth.plan(cluster, lights);
                let own = fourth.action(&sample, &fourth.middle(&plan, r, 0.0, None));
                let given = fourth.action(&sample, &fourth.middle_of(&plan, 0.0, &amplitudes, Some(&fourth.near_exact(&plan, &amplitudes, 1, &fourth.coherent_dressing()))));
                // The generator reads the forms' values alone, the fourth order's part too: of the forms to order 0 it
                // is the same, bit for bit.
                assert_eq!(given, fourth.action(&sample, &fourth.middle_of(&plan, 0.0, &amplitudes, Some(&fourth.near_exact(&plan, &amplitudes, 0, &fourth.coherent_dressing())))), "at {r:?}: the generator of the forms' values alone");
                let lower = second.action(&sample, &second.middle(&second.plan(cluster, lights), r, 0.0, None));
                let size = |x: &[C64]| crate::linalg::norm_sqr(x).sqrt();
                let moved = |x: &[C64], y: &[C64]| size(&x.iter().zip(y).map(|(a, b)| *a - *b).collect::<Vec<_>>());
                assert!(moved(&own, &given) <= 1e-12 * size(&own), "at {r:?}: a middle of given forms {} from the point's own of {}", moved(&own, &given), size(&own));
                assert!(moved(&own, &lower) > 1e-4 * size(&own), "at {r:?}: the fourth order's part is in the middle: {} of {}", moved(&own, &lower), size(&own));
            }
            samples.push(sample);
        }
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == 7 { 1.0 } else { 0.0 });
        let trap = fourth.trap(&reference, &fourth.prepared()).unwrap();
        let reach = fourth.reach(&trap, &reference, 30.0, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        let tables = fourth.tabulate(&reach, 1_000_000);
        let reads = fourth.table_reads(&tables, &samples);
        let mut read = 0;
        for (table, of) in tables.tables.iter().zip(&reads) {
            if table.layout.cluster != cluster {
                continue;
            }
            let whole = of.whole.iter().map(|e| e.population.max(e.temperature)).fold(0.0, f64::max);
            eprintln!("a table of {} axes: its whole error at the samples {whole:.2e}", table.axes());
            assert!(whole <= if table.axes() == 0 { 1e-12 } else { 1e-5 }, "a table of {} axes errs by {whole}", table.axes());
            read += 1;
        }
        assert_eq!(read, 2, "the cluster's tables: under the running lights, and without the trap's");
    }

    /// What the model's part of a step costs at the second order and at the fourth (measured; `SCENE`, default R4_LIN_1):
    /// per cluster of the closed model whose fold has a fourth order, from the two models' tables, one middle at a
    /// thermal point, its path, sixteen generators, eight forces, three tones' forms and the state's scattering -- a
    /// product step's calls --, in µs, with the middle's and the scattering's share; and the tables' reals a node.
    #[test]
    #[ignore]
    fn the_fourth_order_s_cost_a_step() {
        use crate::model::Run;
        use crate::step::Middle as _;
        let name = std::env::var("SCENE").unwrap_or("R4_LIN_1".into());
        let (closed, centre) = closed(&name);
        let grounds: Vec<usize> = (0..closed.clusters.len()).filter(|&c| closed.fold_frequency(c).is_some()).map(|c| closed.clusters[c].states[0]).collect();
        let fourth = closed.rekept(&[closed.kept.clone(), grounds.iter().map(|&ground| Unit::Second { ground }).collect()].concat()).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == grounds[0] { 1.0 } else { 0.0 });
        for (order, model) in [("second", &closed), ("fourth", &fourth)] {
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let reach = model.reach(&trap, &reference, 30.0, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
            let started = std::time::Instant::now();
            let tables = model.tabulate(&reach, 1 << 20);
            let built = started.elapsed();
            let run = Run::tabulated(model, &tables, reference);
            for &ground in &grounds {
                let cluster = model.membership[ground];
                let n = model.clusters[cluster].states.len();
                let psi: Vec<C64> = (0..n).map(|x| C64::cis(0.7 * x as f64) * (1.0 / n as f64).sqrt()).collect();
                let rho: Vec<C64> = (0..n * n).map(|i| psi[i / n] * psi[i % n].conj()).collect();
                let (rounds, mut h, mut sink) = (20_000, Mat::zeros(n, n), 0.0);
                let mut times = [std::time::Duration::ZERO; 7];
                let mut counts = [0.0; 4];
                for round in 0..rounds {
                    let r = [centre[0] + 0.01, centre[1], centre[2] + 0.39 * (round % 97) as f64 / 97.0];
                    let at = Point { r, kappa: [0.3, -0.2, 2.5], t: 0.0 };
                    let from = std::time::Instant::now();
                    let middle = run.middle(cluster, r, 0.0);
                    times[0] += from.elapsed();
                    let path = middle.path(at.kappa, at.kappa.map(|k| k * model.hbar_over_mass));
                    let part = std::time::Instant::now();
                    for node in 0..16 {
                        middle.generator(&path, 0.004 * (node as f64 - 7.5), &mut h);
                        sink += h.data[0].re;
                    }
                    times[3] += part.elapsed();
                    let part = std::time::Instant::now();
                    for node in 0..8 {
                        sink += middle.force(&path, 0.008 * (node as f64 - 3.5), &rho)[2];
                    }
                    times[4] += part.elapsed();
                    middle.tones(&path, &rho, &mut |_, f| sink += f[2]);
                    times[1] += from.elapsed();
                    let work = middle.work();
                    counts = [middle.work, 16.0 * work[0], 8.0 * work[1], 0.0];
                    drop(path);
                    drop(middle);
                    let scattering = std::time::Instant::now();
                    let plan = model.plan(cluster, &model.running());
                    let (scattered, counted) = model.scattering_of(&plan, tables.of(cluster, &model.running()), &at, &psi, None);
                    sink += scattered.force[2];
                    counts[3] = counted;
                    times[2] += scattering.elapsed();
                    let energy = std::time::Instant::now();
                    sink += run.model.energy_with(cluster, &at, &psi, Some(&tables));
                    times[5] += energy.elapsed();
                }
                let each = |t: std::time::Duration| t.as_secs_f64() * 1e6 / rounds as f64;
                let table = tables.of(cluster, &model.running()).map_or(0, |t| t.table.width());
                let ns = |time: std::time::Duration, count: f64| each(time) * 1e3 / count;
                eprintln!(
                    "{name} {order}: cluster of ground state {ground}: ns a counted multiply-add -- middle {:.2} ({:.0}), generators {:.2} ({:.0}), forces {:.2} ({:.0}), scattering {:.2} ({:.0}); an energy {:.2} µs",
                    ns(times[0], counts[0]), counts[0], ns(times[3], counts[1]), counts[1], ns(times[4], counts[2]), counts[2], ns(times[2], counts[3]), counts[3], each(times[5])
                );
                eprintln!("{name} {order}: cluster of ground state {ground} ({n} states): the model's part of a step {:.2} µs (its middle {:.2}, its scattering {:.2}); table {table} reals a node, {} tables in {built:.2?} [{sink:.3e}]", each(times[1]) + each(times[2]), each(times[0]), each(times[2]), tables.tables.len());
            }
        }
    }
}
