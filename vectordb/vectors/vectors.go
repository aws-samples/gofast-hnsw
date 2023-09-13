package vectors

import (
	"math/rand"
)

func GenerateRandomVectors(num int, dimensions int) (vectors [][]float32, err error) {

	vectors = make([][]float32, num)

	for i := 0; i < num; i++ {

		vectors[i] = make([]float32, dimensions)

		for i2 := 0; i2 < dimensions; i2++ {
			vectors[i][i2] = rand.Float32()
		}
	}

	return

}
