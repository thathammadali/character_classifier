"use client";

import { useRef, useState, useEffect } from "react";
import Image from "next/image";

type Classificationresponse = {
  label: string;
  confidence: number;
  explanation: string;
  debug_image: string;
  details: {
    pred_label: number;
    entropy: number;
    style_cluster: number;
    historical_cluster_acc: number;
    historical_cluster_entropy: number;
    rejected: boolean;
  };
};

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [result, setResult] = useState<Classificationresponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"draw" | "upload">("draw");
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  // Initialize canvas
  useEffect(() => {
    if (mode === "draw") {
      clearCanvas();
    }
  }, [mode]);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.beginPath(); // Reset path to avoid connecting lines
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Get coordinates
    let clientX, clientY;
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.lineWidth = 15; // Thick lines for better downscaling
    ctx.lineCap = "round";
    ctx.strokeStyle = "black"; // Black on White (Sketch style)

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
    setResult(null);
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUploadedImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
      submitImage(file);
    }
  };

  const submitDrawing = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Create 28x28 blob
    // Strategy: Draw main canvas onto a small temp canvas
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tCtx = tempCanvas.getContext("2d");
    if (!tCtx) return;

    // Draw uploaded image or canvas?
    // If canvas mode, use canvas. 
    tCtx.drawImage(canvas, 0, 0, 28, 28);

    // Convert to blob
    tempCanvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "drawing.png", { type: "image/png" });
        submitImage(file);
      }
    }, "image/png");
  };

  const submitImage = async (file: File) => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/classify", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server Error: ${res.statusText}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Failed to classify image. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center p-8 bg-zinc-950 text-zinc-100 font-sans">
      <h1 className="text-4xl font-bold mb-2 text-center bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
        Character Classification
      </h1>
      <p className="text-zinc-400 mb-8">Draw a letter or upload an image to see explainable AI in action.</p>

      <div className="flex flex-col lg:flex-row gap-12 w-full max-w-5xl items-start justify-center">

        {/* Left Col: Input */}
        <div className="flex flex-col gap-4 items-center">

          <div className="flex gap-4 mb-2 bg-zinc-900 p-1 rounded-lg border border-zinc-800">
            <button
              onClick={() => setMode("draw")}
              className={`px-4 py-2 rounded-md transition-all ${mode === 'draw' ? 'bg-zinc-700 text-white shadow-md' : 'text-zinc-400 hover:text-white'}`}
            >
              Draw
            </button>
            <button
              onClick={() => setMode("upload")}
              className={`px-4 py-2 rounded-md transition-all ${mode === 'upload' ? 'bg-zinc-700 text-white shadow-md' : 'text-zinc-400 hover:text-white'}`}
            >
              Upload
            </button>
          </div>

          {mode === "draw" ? (
            <div className="relative group">
              <canvas
                ref={canvasRef}
                width={280}
                height={280}
                className="border-2 border-zinc-700 rounded-xl cursor-crosshair shadow-lg shadow-blue-500/10 bg-white touch-none"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
              <div className="flex gap-2 mt-4 justify-center">
                <button
                  onClick={clearCanvas}
                  className="bg-zinc-800 hover:bg-zinc-700 text-white px-6 py-2 rounded-full border border-zinc-700 transition-colors"
                >
                  Clear
                </button>
                <button
                  onClick={submitDrawing}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-2 rounded-full font-medium shadow-lg shadow-blue-900/20 disabled:opacity-50 transition-all hover:scale-105"
                >
                  {loading ? "Analyzing..." : "Classify"}
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4 items-center justify-center w-[280px] h-[280px] border-2 border-dashed border-zinc-700 rounded-xl bg-zinc-900/50">
              <input
                type="file"
                accept="image/*"
                onChange={handleUpload}
                className="hidden"
                id="upload-input"
              />
              <label
                htmlFor="upload-input"
                className="cursor-pointer flex flex-col items-center gap-2 text-zinc-400 hover:text-blue-400 transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" x2="12" y1="3" y2="15" /></svg>
                <span>Click to Upload</span>
              </label>
              {uploadedImage && (
                <div className="absolute w-[280px] h-[280px] flex items-center justify-center pointer-events-none bg-black/80 rounded-xl">
                  <img src={uploadedImage} alt="Preview" className="max-w-full max-h-full object-contain" />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Col: Results */}
        <div className="w-full lg:w-96 flex flex-col gap-6">

          {result ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-zinc-900/80 border border-zinc-800 p-6 rounded-2xl shadow-xl shadow-purple-900/10">
                <div className="flex items-baseline justify-between mb-4 border-b border-zinc-800 pb-4">
                  <span className="text-sm text-zinc-500 uppercase tracking-wider font-semibold">Prediction</span>
                  <span className="text-6xl font-black text-white">{result.label}</span>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium text-zinc-400 mb-1">Confidence</h3>
                    <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-1000 ease-out"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <div className="text-right text-xs text-zinc-500 mt-1">{(result.confidence * 100).toFixed(1)}%</div>
                  </div>

                  <div className="bg-zinc-950/50 p-4 rounded-xl border border-zinc-800/50">
                    <h3 className="text-sm font-medium text-zinc-300 mb-2 flex items-center gap-2">
                      {/* AI Icon */}
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" fill="currentColor" opacity="0.5" /><path d="M12 6a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4z" fill="currentColor" /></svg>
                      Analysis
                    </h3>
                    <p className="text-zinc-400 text-sm leading-relaxed">
                      {result.explanation}
                    </p>
                  </div>

                  {/* Details Grid */}
                  <div className="grid grid-cols-2 gap-4 text-xs text-zinc-500 mt-4 pt-4 border-t border-zinc-800">
                    <div>
                      <span className="block text-zinc-600 mb-1">Entropy</span>
                      <span className="font-mono text-zinc-300">{result.details.entropy.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="block text-zinc-600 mb-1">Style Cluster</span>
                      <span className="font-mono text-zinc-300">#{result.details.style_cluster}</span>
                    </div>
                    <div>
                      <span className="block text-zinc-600 mb-1">Hist. Accuracy</span>
                      <span className="font-mono text-zinc-300">
                        {result.details.historical_cluster_acc >= 0
                          ? `${(result.details.historical_cluster_acc * 100).toFixed(1)}%`
                          : "N/A"}
                      </span>
                    </div>
                  </div>

                  {/* Debug Image */}
                  {result.debug_image && (
                    <div className="mt-6 pt-4 border-t border-zinc-800">
                      <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Model View</h3>
                      <div className="flex items-center gap-4">
                        <div className="relative w-28 h-28 bg-black border border-zinc-700 rounded-lg overflow-hidden shrink-0">
                          <Image
                            src={`data:image/png;base64,${result.debug_image}`}
                            alt="Debug View"
                            fill
                            className="object-contain" // Keep aspect ratio
                            style={{ imageRendering: "pixelated" }} // Show raw pixels
                          />
                        </div>
                        <p className="text-xs text-zinc-500 flex-1">
                          This is exactly what the model "sees" after preprocessing (inversion, cropping, scaling).
                          If this looks wrong (e.g., cut off, inverted colors), the prediction will likely be wrong.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full min-h-[300px] flex items-center justify-center text-zinc-600 bg-zinc-900/20 border border-zinc-800/50 rounded-2xl border-dashed">
              <p className="text-center px-8">
                {loading ? "Processing..." : "Results will appear here"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
