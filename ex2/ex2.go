package ex2

import (
	"fmt"
	"strings"
	"io/ioutil"
	"strconv"
	"github.com/Arafatk/glot"
)

func Ex2() {
	fmt.Println("Ex2")
	denied, admitted := loadScoring()
	
	deniedX := make([]float64, len(denied))
	deniedY := make([]float64, len(denied))
	for d := range denied {
		deniedX[d] = denied[d][0]
		deniedY[d] = denied[d][1]
	}

	admittedX := make([]float64, len(admitted))
	admittedY := make([]float64, len(admitted))
	for d := range admitted {
		admittedX[d] = admitted[d][0]
		admittedY[d] = admitted[d][1]
	}

	dimensions := 2
	persist := false
	debug := false
	plot, _ := glot.NewPlot(dimensions, persist, debug)
	plot.AddPointGroup("denied", "points", [][]float64{deniedX, deniedY})
	plot.AddPointGroup("admitted", "circle", [][]float64{admittedX, admittedY})
	plot.SetXLabel("Exam1 Score")
	plot.SetYLabel("Exam2 Score")
	plot.SavePlot("src/github.com/raphting/machine_learning/plot/ex2.png")
	plot.Close()
	fmt.Println("Done.")
}

func loadScoring() ([][]float64, [][]float64) {
	d, err := ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex2data1.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data := string(d)
	split := strings.Split(data, "\n")
	split = split[:len(split)-1] // ACHTUNG!!!

	denied := make([][]float64, 0)
	admitted := make([][]float64, 0)
	for _, s := range split {
		tmp := strings.Split(s, ",")

		x1, _ := strconv.ParseFloat(tmp[0], 64)
		x2, _ := strconv.ParseFloat(tmp[1], 64)
		pos := []float64{x1, x2}

		// Denied
		if tmp[2] == "0" {
			denied = append(denied, pos)
			continue
		}

		// Admitted
		if tmp[2] == "1" {
			admitted = append(admitted, pos)
		}
	}

	fmt.Println("Denied: ", len(denied))
	fmt.Println("Admitted: ", len(admitted))
	return denied, admitted
}
