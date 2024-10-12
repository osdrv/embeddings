package embeddings

import "math"

func float64ToBytesLE(floats []float64) []byte {
	var bytes []byte
	for _, f := range floats {
		bits := math.Float64bits(f)
		bytes = append(bytes,
			byte(bits), // Least significant byte first
			byte(bits>>8),
			byte(bits>>16),
			byte(bits>>24),
			byte(bits>>32),
			byte(bits>>40),
			byte(bits>>48),
			byte(bits>>56), // Most significant byte last
		)
	}
	return bytes
}
