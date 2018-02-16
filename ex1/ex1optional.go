package main

import (
	"gonum.org/v1/gonum/stat"
	"fmt"
	"io/ioutil"
	"strings"
	"strconv"
	"gonum.org/v1/gonum/mat"
)

func optionalTasks() {
	// Load data
	size, beds, price := loadData2()
	// Normalize data
	size = normalize(size)
	beds = normalize(beds)
	price = normalize(price)

	m := len(size)
	ones := make([]float64, m)
	for k := range ones {
		ones[k] = 1
	}

	// Create Feature Vector including ones
	X := mat.NewDense(m, 3, nil)
	X.SetCol(0, ones)
	X.SetCol(1, size)
	X.SetCol(2, beds)
	// Create trained data
	y := mat.NewDense(m, 1, price)
	// Create start values for theta
	theta := mat.NewDense(3, 1, []float64{0, 0, 0})

	iterations := 1500
	alpha := 0.01
	theta = gradientDescent(X, y, theta, alpha, iterations)

	fc := mat.Formatted(theta, mat.Prefix("             "), mat.Squeeze())
	fmt.Printf("Regression = %v\n", fc)
}

func normalize(data []float64) []float64{
	// Get Standard Deviation
	stddev := stat.StdDev(data, nil)
	// Calculate Average
	avg := float64(0)
	for x := range data {
		avg += data[x]
	}
	avg = avg / float64(len(data))

	// normalize data
	for x := range data {
		data[x] = (data[x] - avg) / stddev
	}
	return data
}

func loadData2() ([]float64, []float64, []float64) {
	d, err := ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex1data2.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data := string(d)
	split := strings.Split(data, "\n")
	split = split[:len(split)-1] // ACHTUNG!!!

	size := make([]float64, len(split))
	beds := make([]float64, len(split))
	price := make([]float64, len(split))
	for k, s := range split {
		tmp := strings.Split(s, ",")
		i, _ := strconv.ParseFloat(tmp[0], 64)
		size[k] = float64(i)

		i, _ = strconv.ParseFloat(tmp[1], 64)
		beds[k] = float64(i)

		i, _ = strconv.ParseFloat(tmp[2], 64)
		price[k] = float64(i)
	}

	return size, beds, price
}
