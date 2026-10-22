fn main() {
    let values: Vec<u64> = (1..=100).collect();
    let sum: u64 = std::hint::black_box(values).iter().sum();
    assert_eq!(sum, 5050);
    println!("zlc-v3 Rust release smoke passed: sum={sum}");
}
