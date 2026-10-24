//! Append-only Q^T A Q projection for ONE fixed operator and basis sequence.
//! Stores only the small projected matrix. No physical solution, model,
//! trajectory, or large Q/AQ prefix is stored or hashed.
use crate::Lapack;

#[derive(Default)]
pub struct Projection {
    rows: Option<usize>,
    size: usize,
    projected: Vec<f64>,
}

impl Projection {
    pub fn new()->Self { Self::default() }

    /// Required before changing the operator, row coordinates, or any old
    /// basis/product column. A fresh trajectory should use a fresh instance.
    pub fn clear(&mut self) {
        self.rows=None;
        self.size=0;
        self.projected.clear();
    }

    /// `columns` and `products` are n-by-m column-major Q and A*Q.
    /// The caller guarantees their existing prefixes never change. We check
    /// dimensions and monotonic m, NOT the values of an already used prefix.
    /// For delta new columns compute only Q_all^T AQ_new and Q_new^T AQ_old.
    /// A need not be symmetric; the two off-diagonal blocks are independent.
    pub fn update(
        &mut self,lap:&Lapack,columns:&[f64],products:&[f64],rows:usize,size:usize,
    )->Result<Vec<f64>,String>{
        if rows==0||rows>i32::MAX as usize||size>i32::MAX as usize
            ||rows.checked_mul(size)!=Some(columns.len())
            ||rows.checked_mul(size)!=Some(products.len())
            ||size.checked_mul(size).is_none()
            ||self.rows.is_some_and(|old|old!=rows)||size<self.size
            ||self.size.checked_mul(self.size)!=Some(self.projected.len()){
            return Err("Invalid append-only projection dimensions; clear before changing operator or basis".into());
        }
        if size==self.size {
            self.rows=Some(rows);
            return Ok(self.projected.clone());
        }
        let old=self.size;
        let delta=size-old;
        let mut next=vec![0.;size*size];
        for j in 0..old {
            next[j*size..j*size+old].copy_from_slice(&self.projected[j*old..(j+1)*old]);
        }
        // All rows / new columns, including the new-new corner. Inputs are
        // contiguous column views into the existing Q and AQ allocations.
        lap.multiply_transpose_left_into(
            columns,&products[rows*old..],&mut next[size*old..],
            rows,size,delta,rows,rows,size,
        )?;
        if old>0 {
            // New rows / old columns. ldc=size skips the copied top block;
            // the last addressed element ends exactly at column old-1.
            lap.multiply_transpose_left_into(
                &columns[rows*old..],&products[..rows*old],&mut next[old..size*old],
                rows,delta,old,rows,rows,size,
            )?;
        }
        if next.iter().any(|v|!v.is_finite()) {
            return Err("Non-finite incremental projected operator".into());
        }
        // Commit only after both BLAS operations and the small-result check
        // succeed. Invalid calls cannot partially replace a usable prefix.
        self.rows=Some(rows);
        self.size=size;
        self.projected=next;
        Ok(self.projected.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(lap:&Lapack,n:usize,m:usize,offset:f64)->(Vec<f64>,Vec<f64>){
        let a=(0..n*n).map(|index|{
            let(i,j)=(index%n,index/n);
            ((i*7+j*3)%13)as f64/7.+if i==j{2.+offset}else{offset*(i+1)as f64/(j+1)as f64}
        }).collect::<Vec<_>>();
        let q=(0..n*m).map(|index|{
            let(i,j)=(index%n,index/n);
            ((i+1)as f64*(j as f64+0.3)).sin()+0.1*(i+j)as f64
        }).collect::<Vec<_>>();
        let aq=lap.multiply(&a,&q,n,m,n);
        (q,aq)
    }
    fn full(lap:&Lapack,q:&[f64],aq:&[f64],n:usize,m:usize)->Vec<f64>{
        let qt=(0..n).flat_map(|i|(0..m).map(move|j|q[i+j*n])).collect::<Vec<_>>();
        lap.multiply(&qt,aq,m,m,n)
    }
    fn close(a:&[f64],b:&[f64]){
        assert_eq!(a.len(),b.len());
        for(&a,&b)in a.iter().zip(b){assert!((a-b).abs()<2e-12*(1.+b.abs()),"{a} versus {b}");}
    }

    #[test]
    fn incremental_projection_nonsymmetric_four_blocks_match_full(){
        let lap=Lapack::load().unwrap();let(n,m)=(7,5);let(q,aq)=fixture(&lap,n,m,0.2);
        let mut projection=Projection::new();let mut old=Vec::<f64>::new();let mut old_size=0;
        for size in [1,2,4,5,5] {
            let actual=projection.update(&lap,&q[..n*size],&aq[..n*size],n,size).unwrap();
            close(&actual,&full(&lap,&q[..n*size],&aq[..n*size],n,size));
            for j in 0..old_size{for i in 0..old_size{assert_eq!(actual[i+j*size].to_bits(),old[i+j*old_size].to_bits());}}
            old=actual;old_size=size;
        }
        assert!((old[1]-old[m]).abs()>1e-3,"fixture must distinguish both off-diagonal blocks");
    }

    #[test]
    fn incremental_projection_reset_and_operator_scope_are_local(){
        let lap=Lapack::load().unwrap();let(n,m)=(5,3);let(q,a)=fixture(&lap,n,m,0.);
        let(_,b)=fixture(&lap,n,m,1.7);let mut first=Projection::new();let mut second=Projection::new();
        let first_value=first.update(&lap,&q,&a,n,m).unwrap();let second_value=second.update(&lap,&q,&b,n,m).unwrap();
        assert!(first_value.iter().zip(&second_value).any(|(a,b)|(a-b).abs()>1.));
        first.clear();close(&first.update(&lap,&q,&b,n,m).unwrap(),&second_value);
        first.clear();let(q,a)=fixture(&lap,3,2,0.4);
        close(&first.update(&lap,&q,&a,3,2).unwrap(),&full(&lap,&q,&a,3,2));
    }

    #[test]
    fn incremental_projection_rejects_invalid_shapes_and_old_size(){
        let lap=Lapack::load().unwrap();let(q,aq)=fixture(&lap,5,3,0.2);let mut projection=Projection::new();
        let valid=projection.update(&lap,&q,&aq,5,3).unwrap();
        assert!(projection.update(&lap,&q[..14],&aq,5,3).is_err());
        assert!(projection.update(&lap,&q,&aq[..14],5,3).is_err());
        assert!(projection.update(&lap,&q[..10],&aq[..10],5,2).is_err());
        assert!(projection.update(&lap,&q[..12],&aq[..12],4,3).is_err());
        assert!(projection.update(&lap,&[],&[],0,0).is_err());
        assert!(projection.update(&lap,&[],&[],usize::MAX,usize::MAX).is_err());
        assert_eq!(projection.update(&lap,&q,&aq,5,3).unwrap(),valid);
        // Padded leading dimensions and a strided output subblock are real
        // BLAS layouts. Padding must survive; invalid spans must touch nothing.
        let mut out=vec![123.;5];
        lap.multiply_transpose_left_into(&[1.,2.,99.,3.,4.],&[5.,6.,99.,99.,7.,8.],&mut out,2,2,2,3,4,3).unwrap();
        assert_eq!(out,[17.,39.,123.,23.,53.]);let saved=out.clone();
        assert!(lap.multiply_transpose_left_into(&[1.;4],&[1.;4],&mut out,2,2,2,1,2,3).is_err());
        assert!(lap.multiply_transpose_left_into(&[1.;4],&[1.;4],&mut out,2,2,2,2,2,2).is_err());
        assert!(lap.multiply_transpose_left_into(&[],&[],&mut out,usize::MAX,2,2,2,2,3).is_err());
        assert_eq!(out,saved);
    }
}
