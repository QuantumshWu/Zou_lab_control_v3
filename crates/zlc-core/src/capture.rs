//! Capture evidence validation at batch and terminal boundaries.
//!
//! Drivers own their FIFO, synchronization, SDK thread and buffer lifetimes.
//! These typed functions validate facts presented at publication boundaries;
//! they own no hardware queue and require no per-record foreign-function call.

#[derive(Clone,Debug,Eq,PartialEq)]
pub enum CaptureError {
    ZeroGroupSize,
    OrdinalGap { expected:u64, observed:u64 },
    OrdinalOverflow,
    IncompleteGroup { expected_start:u64, expected_count:u64, observed_count:usize },
    IncompleteTerminal,
    TerminalCount { expected:u64, produced:u64 },
}
impl std::fmt::Display for CaptureError {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        match self {
            Self::ZeroGroupSize=>write!(f,"capture group size must be positive"),
            Self::OrdinalGap{expected,observed}=>write!(f,"record sequence gap: expected {expected}, received {observed}"),
            Self::OrdinalOverflow=>write!(f,"capture record ordinal overflow"),
            Self::IncompleteGroup{expected_start,expected_count,observed_count}=>write!(f,"cycle source ordinals are not contiguous: expected {expected_count} from {expected_start}, received {observed_count} records"),
            Self::IncompleteTerminal=>write!(f,"terminal evidence is incomplete: the capture did not stop and join"),
            Self::TerminalCount{expected,produced}=>write!(f,"terminal count differs from completed cycles: expected {expected}, produced {produced} (a partial cycle may be present)"),
        }
    }
}
impl std::error::Error for CaptureError {}

#[derive(Clone,Debug,Eq,PartialEq)]
pub struct BatchEvidence {pub next:u64,pub group_ends:Vec<u64>}

/// Validate the entire batch before returning any new sequence/group evidence.
pub fn validate_batch(expected_start:u64,ordinals:&[u64],group_size:u64)->Result<BatchEvidence,CaptureError>{
    if group_size==0{return Err(CaptureError::ZeroGroupSize);}
    let mut next=expected_start;let mut group_ends=Vec::new();
    for &observed in ordinals {
        if observed!=next{return Err(CaptureError::OrdinalGap{expected:next,observed});}
        next=next.checked_add(1).ok_or(CaptureError::OrdinalOverflow)?;
        if next%group_size==0{group_ends.push(observed);}
    }
    Ok(BatchEvidence{next,group_ends})
}

pub fn validate_group(expected_start:u64,ordinals:&[u64],group_size:u64)->Result<(),CaptureError>{
    if group_size==0{return Err(CaptureError::ZeroGroupSize);}
    if expected_start%group_size!=0 || ordinals.len() as u64!=group_size {
        return Err(CaptureError::IncompleteGroup{expected_start,expected_count:group_size,observed_count:ordinals.len()});
    }
    validate_batch(expected_start,ordinals,group_size)?;Ok(())
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct TerminalEvidence {pub produced_count:u64,pub source_stopped:bool,pub joined:bool}
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub struct TerminalSummary {pub produced:u64,pub complete_groups:u64,pub partial_records:u64}

pub fn validate_terminal(proof:TerminalEvidence,expected:Option<u64>,stopped:bool,group_size:u64)->Result<TerminalSummary,CaptureError>{
    if !proof.source_stopped || !proof.joined{return Err(CaptureError::IncompleteTerminal);}
    if group_size==0{return Err(CaptureError::ZeroGroupSize);}
    if let Some(expected)=expected {
        if proof.produced_count<expected || (proof.produced_count!=expected && !stopped) {
            return Err(CaptureError::TerminalCount{expected,produced:proof.produced_count});
        }
    }
    Ok(TerminalSummary{produced:proof.produced_count,complete_groups:proof.produced_count/group_size,partial_records:proof.produced_count%group_size})
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn batch_boundaries_close_groups_across_split_reads(){
        let first=validate_batch(0,&[0,1],3).unwrap();assert_eq!(first,BatchEvidence{next:2,group_ends:vec![]});
        let second=validate_batch(first.next,&[2,3,4,5,6],3).unwrap();assert_eq!(second.group_ends,vec![2,5]);assert_eq!(second.next,7);
        assert_eq!(validate_batch(7,&[],3).unwrap().next,7);
    }
    #[test]fn invalid_batches_never_produce_successful_evidence(){
        assert_eq!(validate_batch(4,&[4,6],2),Err(CaptureError::OrdinalGap{expected:5,observed:6}));
        assert_eq!(validate_batch(u64::MAX,&[u64::MAX],1),Err(CaptureError::OrdinalOverflow));
        assert_eq!(validate_batch(0,&[0],0),Err(CaptureError::ZeroGroupSize));
        assert!(validate_group(4,&[4],2).is_err());assert!(validate_group(3,&[3,4],2).is_err());validate_group(4,&[4,5],2).unwrap();
    }
    #[test]fn terminal_proofs_keep_partial_cycles_distinct(){
        let proof=TerminalEvidence{produced_count:5,source_stopped:true,joined:true};
        assert!(validate_terminal(proof,Some(4),false,2).is_err());
        assert_eq!(validate_terminal(proof,Some(4),true,2).unwrap(),TerminalSummary{produced:5,complete_groups:2,partial_records:1});
        assert!(validate_terminal(proof,Some(6),true,2).is_err());
        assert!(validate_terminal(TerminalEvidence{joined:false,..proof},None,true,2).is_err());
    }
}
