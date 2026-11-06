//! A warm plugin must not reuse an accumulator from another native service.
use serde_json::{Value,json};
use super::{Service,Tensor};

fn ask(service:&mut Service,operation:&str,args:&Value)->Result<Value,String>{
    service.dispatch(operation,args,&|_|Err("unexpected array read".into()),&|_:Tensor|Err("unexpected array write".into()))
}

fn create(service:&mut Service)->String{
    let response=ask(service,"scientific.group.create",&json!({
        "statistic":"mean","domain":0,"spanning":true,"every":2,
        "counts_r":[1],"counts_p":[1],"source_points":1,"output_points":1,
        "lines":null,"steps":null,
    })).unwrap();
    response["handle"].as_str().unwrap().to_owned()
}

#[test]
fn old_run_handles_cannot_read_feed_finish_or_drop_a_new_runs_group(){
    let mut old=Service::default();
    let mut new=Service::default();
    let stale=create(&mut old);
    let current=create(&mut new);
    assert_ne!(stale,current,"first handles of separate services must not alias");
    assert!(uuid::Uuid::parse_str(&stale).is_ok());
    assert!(uuid::Uuid::parse_str(&current).is_ok());
    for operation in ["scientific.group.feed","scientific.group.state","scientific.group.finish"]{
        let error=ask(&mut new,operation,&json!({"handle":stale})).unwrap_err();
        assert!(error.contains("not current"),"{operation}: {error}");
        let error=ask(&mut old,operation,&json!({"handle":current})).unwrap_err();
        assert!(error.contains("not current"),"{operation}: {error}");
    }
    // Drop is deliberately idempotent, but cannot remove another run's group.
    ask(&mut new,"scientific.group.drop",&json!({"handle":stale})).unwrap();
    assert_eq!(ask(&mut new,"scientific.group.state",&json!({"handle":current})).unwrap()["open_groups"],0);
    assert!(ask(&mut old,"scientific.group.state",&json!({"handle":stale})).is_ok());
    assert_eq!(ask(&mut new,"scientific.group.finish",&json!({"handle":current})).unwrap()["chunks"],json!([]));
    assert!(ask(&mut new,"scientific.group.state",&json!({"handle":current})).is_err());
    let replacement=create(&mut new);
    assert_ne!(replacement,current);
    assert!(ask(&mut new,"scientific.group.feed",&json!({"handle":current})).unwrap_err().contains("not current"));
}
