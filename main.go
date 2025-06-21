package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"syscall"
	"time"
	"unsafe"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
)

var (
	dll         = syscall.NewLazyDLL("onnx_model_infer_win64_model.dll")
	procInit    = dll.NewProc("ONNX_MODEL_INFER_Init")
	procPredict = dll.NewProc("ONNX_MODEL_INFER_Predict")
	procCleanup = dll.NewProc("ONNX_MODEL_INFER_Cleanup")
	labels      = []string{
		"Cardboard", "Food Organics", "Glass", "Metal",
		"Miscellaneous Trash", "Paper", "Plastic",
		"Textile Trash", "Vegetation",
	}
)

type PredictionResponse struct {
	ClassIndex int32  `json:"class_index"`
	Label      string `json:"label"`
	TimeMs     int64  `json:"time_ms"`
}

func toTensorCHW(img image.Image) []float32 {
	const W, H = 128, 128
	// nearest‑resize + CHW + normalize to [-1,1]
	tensor := make([]float32, 3*W*H)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			px := img.At(x*img.Bounds().Dx()/W, y*img.Bounds().Dy()/H)
			r, g, b, _ := px.RGBA()
			idx := y*W + x
			norm := func(v uint32) float32 { return (float32(v>>8)/255.0 - 0.5) / 0.5 }
			tensor[idx] = norm(r)
			tensor[idx+W*H] = norm(g)
			tensor[idx+2*W*H] = norm(b)
		}
	}
	return tensor
}

func initModel() error {
	dll.Load()
	modelPath, _ := syscall.UTF16PtrFromString("model.onnx")
	ret, _, _ := procInit.Call(uintptr(unsafe.Pointer(modelPath)))
	if ret != 0 {
		return fmt.Errorf("failed to initialize model")
	}
	return nil
}

func predictImage(img image.Image) (*PredictionResponse, error) {
	input := toTensorCHW(img)
	var classIdx int32
	start := time.Now()

	ret, _, _ := procPredict.Call(
		uintptr(unsafe.Pointer(&input[0])),
		uintptr(len(input)),
		uintptr(unsafe.Pointer(&classIdx)),
	)

	if ret != 0 {
		return nil, fmt.Errorf("prediction failed")
	}

	return &PredictionResponse{
		ClassIndex: classIdx,
		Label:      labels[classIdx],
		TimeMs:     time.Since(start).Milliseconds(),
	}, nil
}

func main() {
	if err := initModel(); err != nil {
		panic(err)
	}
	defer procCleanup.Call()

	app := fiber.New()
	app.Use(cors.New())

	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status": "ok",
		})
	})

	app.Post("/predict", func(c *fiber.Ctx) error {

		file, err := c.FormFile("image")
		if err != nil {
			return c.Status(400).JSON(fiber.Map{
				"error": "Please upload an image file",
			})
		}

		fileContent, err := file.Open()
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error": "Failed to process image",
			})
		}
		defer fileContent.Close()

		img, _, err := image.Decode(fileContent)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{
				"error": "Invalid image format",
			})
		}

		prediction, err := predictImage(img)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error": err.Error(),
			})
		}

		return c.JSON(prediction)
	})

	app.Listen(":3000")
}
