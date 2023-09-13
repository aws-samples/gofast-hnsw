package distance_test

import (
	"fmt"
	"testing"

	"github.com/aws-samples/gofast-hnsw/vectordb/distance"
	"github.com/aws-samples/gofast-hnsw/vectordb/vectors"
	"github.com/stretchr/testify/assert"
)

// Test vars
type TestCases struct {
	x        []float32
	y        []float32
	distance float32
}

var vectorTest = []TestCases{
	{
		x:        []float32{0.89134955, 0.076961525, 0.2982818, 0.39917126, 0.61602664, 0.8120031, 0.0380205, 0.2586385, 0.89134955, 0.076961525, 0.2982818, 0.39917126, 0.61602664, 0.8120031, 0.0380205, 0.2586385},
		y:        []float32{0.44481596, 0.54167795, 0.7662333, 0.9859179, 0.17665349, 0.40175834, 0.49756145, 0.98521996, 0.44481596, 0.54167795, 0.7662333, 0.9859179, 0.17665349, 0.40175834, 0.49756145, 0.98521996},
		distance: 4.158104,
	},

	{
		x:        []float32{1, 2, 3, 4},
		y:        []float32{5, 6, 7, 8},
		distance: 64,
	},

	{
		x:        []float32{1.039, 2.203, 3.203, 4.20},
		y:        []float32{5.303, 6.203, 7.209, 8.9},
		distance: 72.319725,
	},

	{
		x: []float32{0.9777251, 0.13124035, 0.19925745, 0.6260118, 0.65496475, 0.847665, 0.15405779, 0.30328768,
			0.6407562, 0.8027169, 0.8273791, 0.28925145, 0.32593518, 0.75363505, 0.81362045, 0.97445536,
			0.40187705, 0.46209162, 0.88369006, 0.92534506, 0.9854233, 0.40738583, 0.5704553, 0.8702612,
			0.055267714, 0.36209768, 0.51987344, 0.5787446, 0.010325488, 0.043180007, 0.50580496, 0.53382087},

		y: []float32{0.691518, 0.02940946, 0.1829301, 0.60970354, 0.6774755, 0.10152412, 0.13004503, 0.20111531,
			0.6246105, 0.77858824, 0.11370886, 0.26739648, 0.5740263, 0.76276165, 0.79153717, 0.22237052,
			0.28617415, 0.7176004, 0.7463197, 0.9345657, 0.24118538, 0.39465192, 0.46276712, 0.88933474,
			0.9181098, 0.37321398, 0.41352624, 0.83634037, 0.8964462, 0.06211478, 0.48568678, 0.5449081},
		distance: 4.0714846,
	},
}

func Test_L2(t *testing.T) {

	for i := range vectorTest {
		d1, _ := distance.L2_1x(vectorTest[i].x, vectorTest[i].y)

		assert.Equal(t, d1, vectorTest[i].distance)

		fmt.Println(vectorTest[i].x, vectorTest[i].y)

	}

}

func Test_L2_WrongLen(t *testing.T) {

	v := []float32{1, 2}
	v2 := []float32{1, 2, 3, 4}
	_, err := distance.L2_Opt(&v, &v2)

	assert.Nil(t, err)

}

func Benchmark_L2_1x(b *testing.B) {

	vec, _ := vectors.GenerateRandomVectors(2, 8)

	for n := 0; n < b.N; n++ {
		_, _ = distance.L2_1x(vec[0], vec[1])
	}

}

func Benchmark_L2_Opt(b *testing.B) {

	vec, _ := vectors.GenerateRandomVectors(2, 8)

	for n := 0; n < b.N; n++ {
		_, _ = distance.L2_Opt(&vec[0], &vec[1])
	}

}

func Benchmark_Large_L2_1x(b *testing.B) {

	vec, _ := vectors.GenerateRandomVectors(2, 1024)

	for n := 0; n < b.N; n++ {
		_, _ = distance.L2_Opt(&vec[0], &vec[1])
	}

}

func Benchmark_Large_L2_1x_Opt(b *testing.B) {

	vec, _ := vectors.GenerateRandomVectors(2, 1024)

	for n := 0; n < b.N; n++ {
		_, _ = distance.L2_Opt(&vec[0], &vec[1])
	}

}
