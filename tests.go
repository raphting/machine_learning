package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func main() {
	A := mat.NewDense(3, 2, []float64{
		1, 3,
		2, 4,
		0, 5,
	})
	B := mat.NewDense(2, 2, []float64{
		1, 0,
		2, 3,
	})

	var C mat.Dense

	C.Mul(A, B)
	fc := mat.Formatted(&C, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v\n", fc)
}
