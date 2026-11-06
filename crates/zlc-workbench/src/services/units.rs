//! Language-neutral adapters for the core's unit registry and conversions.
use std::sync::Arc;
use serde_json::{Value,json};
use zlc_core::units::{Unit,UnitRegistry,Conversion,DEFAULT_UNITS};
use zlc_core::decimal::Decimal;
use zlc_core::scientific::Tensor;
type Result<T> = std::result::Result<T,String>;

fn number(v:&Value,name:&str)->Result<f64> {
    v[name].as_f64().filter(|v|v.is_finite()).ok_or_else(||format!("unit {name} must be finite"))
}
fn conversion(v:&Value)->Result<Conversion> {
    Ok(match v["kind"].as_str().unwrap_or("") {
        "scaled"=>Conversion::scaled(number(v,"factor")?).map_err(|e|e.to_string())?,
        "offset"=>Conversion::Offset(number(v,"offset")?),
        "decibel"=> {let(reference,factor)=(number(v,"reference")?,number(v,"factor")?);
            if reference<=0.0||factor==0.0{return Err("decibel reference must be positive and factor nonzero".into());} Conversion::Decibel{reference,factor}},
        "voltage"=> {let(load_ohms,divisor)=(number(v,"load_ohms")?,number(v,"divisor")?);
            if load_ohms<=0.0||divisor<=0.0{return Err("voltage load and mean-square divisor must be positive".into());} Conversion::VoltageIntoLoad{load_ohms,divisor}},
        "prefixed"=> {let factor=number(v,"factor")?;if factor<=0.0{return Err("prefix factor must be positive".into());}
            Conversion::Prefixed{inner:Box::new(conversion(&v["inner"])?),factor}},
        _=>return Err("unknown unit conversion".into()),
    })
}
fn conversion_tree(v:&Conversion)->Value {
    match v {
        Conversion::Scaled(factor)=>json!({"kind":"scaled","factor":factor}),
        Conversion::Offset(offset)=>json!({"kind":"offset","offset":offset}),
        Conversion::Decibel{reference,factor}=>json!({"kind":"decibel","reference":reference,"factor":factor}),
        Conversion::VoltageIntoLoad{load_ohms,divisor}=>json!({"kind":"voltage","load_ohms":load_ohms,"divisor":divisor}),
        Conversion::Prefixed{inner,factor}=>json!({"kind":"prefixed","inner":conversion_tree(inner),"factor":factor}),
    }
}
fn unit(v:&Value)->Result<Unit> {
    let texts=|name:&str|->Result<Vec<String>> {serde_json::from_value(v.get(name).cloned().unwrap_or_else(||json!([]))).map_err(|e|e.to_string())};
    let(prefixes,aliases)=(texts("prefixes")?,texts("aliases")?);
    Unit::new(v["symbol"].as_str().ok_or("unit symbol must be text")?,v["dimension"].as_str().ok_or("unit dimension must be text")?,
        conversion(&v["conversion"])?,&prefixes.iter().map(String::as_str).collect::<Vec<_>>(),&aliases.iter().map(String::as_str).collect::<Vec<_>>(),
        v["inverse_dimension"].as_str()).map_err(|e|e.to_string())
}
fn tree(unit:&Unit)->Value {
    json!({"symbol":unit.symbol,"dimension":unit.dimension,"conversion":conversion_tree(&unit.conversion),
        "prefixes":unit.prefixes,"aliases":unit.aliases,"inverse_dimension":unit.inverse_dimension,"decade":unit.decade,
        "linear":unit.is_linear(),"base":unit.is_base(),"coordinate_scale":unit.coordinate_scale(),
        "ladder":unit.ladder().map(|p|p.symbol).collect::<Vec<_>>()})
}

pub fn dispatch(op:&str,args:&Value,read:&dyn Fn(&Value)->Result<Tensor>,write:&dyn Fn(Tensor)->Result<Value>)->Result<Value> {
    if op=="units.prefixes" {return Ok(json!(zlc_core::units::PREFIXES.iter().map(|p|json!({"symbol":p.symbol,"exponent":p.exponent,"accepts":p.accepts})).collect::<Vec<_>>()));}
    if op=="units.conversion_facts" {
        let c=conversion(&args["conversion"])?;
        return Ok(json!({"conversion":conversion_tree(&c),"initial_real_to":c.initial_real_input(false),"initial_real_from":c.initial_real_input(true)}));
    }
    if op=="units.builtins" {return Ok(json!({"units":DEFAULT_UNITS.definitions().iter().map(|u|tree(u)).collect::<Vec<_>>()}));}
    if op=="units.validate"||op=="units.facts" {return unit(&args["unit"]).map(|unit|tree(&unit));}
    if op=="units.apply"||op=="units.convert" {
        let input=read(&args["values"])?;input.validate()?;
        let result=if op=="units.apply" {
            let conversion=conversion(&args["conversion"])?;
            conversion.apply_array(&input.values,args["inverse"]==true)
        } else {
            let(origin,target)=(unit(&args["source"])?,unit(&args["target"])?);
            origin.convert_array_to(&input.values,&target).map_err(|e|e.to_string())?
        };
        return Ok(json!({"values":write(Tensor{values:result.values,shape:input.shape})?,"discarded_imaginary":result.discarded_imaginary}));
    }
    let definitions:Vec<Unit>=args["units"].as_array().ok_or("unit registry needs its definitions")?.iter().map(unit).collect::<Result<_>>()?;
    let mut registry=UnitRegistry::new(definitions).map_err(|e|e.to_string())?;
    if op=="units.register" {
        registry.register(unit(&args["unit"])?,args["replace"]==true).map_err(|e|e.to_string())?;
        return Ok(json!({"units":registry.definitions().iter().map(|u|tree(u)).collect::<Vec<_>>()}));
    }
    let resolve=|v:&Value|->Result<Arc<Unit>> { if let Some(text)=v.as_str(){registry.resolve(text).map_err(|e|e.to_string())}else{unit(v).map(Arc::new)} };
    let answer=match op {
        "units.parse"=>{
            let u=resolve(&args["unit"])?;
            let number=zlc_core::units::parse_quantity(args["text"].as_str().ok_or("a typed quantity must be text")?,&u,Some(&registry)).map_err(|e|e.to_string())?;
            if number.is_finite(){json!(number)}else{json!({"$float":zlc_core::pyfmt::repr(number)})}
        },
        "units.prefix"=>{
            let u=resolve(&args["unit"])?;
            let values=args["values"].as_array().ok_or("prefix values must be decimal texts")?.iter().map(|v|v.as_str().and_then(Decimal::parse).ok_or("prefix value is not a decimal".to_owned())).collect::<Result<Vec<_>>>()?;
            json!(zlc_core::units::prefix_for_decimals(&values,&u).exponent)
        },
        "units.format"=>{
            let u=resolve(&args["unit"])?;let value=Decimal::parse(args["value"].as_str().ok_or("quantity value must be decimal text")?).ok_or("quantity value is not decimal")?;
            let prefix=if args["prefix"].is_null(){None}else{
                Some(zlc_core::units::PREFIXES.iter().find(|p|args["prefix"]["symbol"]==p.symbol&&args["prefix"]["exponent"]==p.exponent
                    &&args["prefix"]["accepts"]==json!(p.accepts)).ok_or("quantity prefix is not a canonical prefix")?)
            };
            json!(zlc_core::units::format_decimal_quantity(&value,&u,prefix).map_err(|e|e.to_string())?)
        },
        "units.resolve"=>tree(resolve(&args["unit"])?.as_ref()),
        "units.base"=>{
            let base=if let Some(text)=args["unit"].as_str(){registry.base_for(text).map_err(|e|e.to_string())?}
                else{registry.base_for_unit(resolve(&args["unit"])?.as_ref()).map_err(|e|e.to_string())?};
            tree(base.as_ref())
        },
        "units.family"=>{let u=resolve(&args["unit"])?;let(base,prefix)=registry.family_and_prefix(&u).map_err(|e|e.to_string())?;
            json!({"unit":tree(&base),"prefix":prefix.exponent})},
        "units.inverse"=>registry.inverse_for(resolve(&args["unit"])?.as_ref()).map_or(Value::Null,|u|tree(&u)),
        "units.inverse_symbol"=>json!(registry.inverse_symbol(resolve(&args["unit"])?.as_ref())),
        "units.choices"=>json!(registry.display_choices(resolve(&args["unit"])?.as_ref()).map_err(|e|e.to_string())?),
        "units.decimal"=> {
            let number=Decimal::parse(args["value"].as_str().ok_or("decimal value must be text")?).ok_or("unit conversion requires a finite scalar")?;
            let converted=registry.convert_decimal(&number,resolve(&args["source"])?.as_ref(),resolve(&args["target"])?.as_ref()).map_err(|e|e.to_string())?;
            json!(converted.to_sci_string())
        },
        _=>return Err(format!("unknown unit operation {op}")),
    };
    Ok(answer)
}
