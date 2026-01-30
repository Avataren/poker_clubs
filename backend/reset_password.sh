#!/bin/bash
# Generate a bcrypt hash for "password123"
RUST_CODE='
use bcrypt;
fn main() {
    let hash = bcrypt::hash("password123".as_bytes(), bcrypt::DEFAULT_COST).unwrap();
    println!("{}", hash);
}
'
echo "$RUST_CODE" | cargo script -
