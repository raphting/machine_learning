package ex2

import (
	"fmt"
	"github.com/cdipaolo/goml/linear"
	"github.com/cdipaolo/goml/base"
)

func Ex2() {
	exams()
	fmt.Println("\n\n")
	chips()
}

func exams() {
	x, y, err := base.LoadDataFromCSV("src/github.com/raphting/machine_learning/data/ex2data1.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	logistic := linear.NewLogistic(base.BatchGA, 0.001, 6,1000, x, y)
	err = logistic.Learn()
	if err != nil {
		fmt.Println(err.Error())
	}


	// Prediction
	pred := []float64{45, 85}
	p, err := logistic.Predict(pred)
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Printf("Scoring 45, 85 predicts to: %f\n", p[0])

	testClassifier(x, y, logistic)
}

func chips() {
	x, y, err := base.LoadDataFromCSV("src/github.com/raphting/machine_learning/data/ex2data2.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	logistic := linear.NewLogistic(base.BatchGA, 0.0001, 6, 10000, x, y)
	err = logistic.Learn()
	if err != nil {
		fmt.Println(err.Error())
	}


	// Prediction
	pred := []float64{-0.25, 1.5}

	p, err := logistic.Predict(pred)
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Printf("Chip with -0.25, 1.5 classifies to: %f\n", p[0])

	testClassifier(x, y, logistic)
}

func testClassifier(x [][]float64, y []float64, logistic *linear.Logistic) {
	// Test prediction
	correct := 0
	wrong := 0
	for t := range x {
		p, _ := logistic.Predict([]float64{x[t][0], x[t][1]})
		if y[t] == 1 && p[0] >= float64(0.5) {
			correct++
			continue
		}

		if y[t] == 0 && p[0] < float64(0.5) {
			correct++
			continue
		}

		wrong++
	}

	fmt.Println("Test classification on existing data: Correct: ", correct, " Wrong: ", wrong)
}
