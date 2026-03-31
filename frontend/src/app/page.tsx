"use client";

import { useState, useCallback } from "react";
import VideoUpload from "@/components/VideoUpload";
import ProcessingStatus from "@/components/ProcessingStatus";
import CourtHeatmap from "@/components/CourtHeatmap";
import RallyStats from "@/components/RallyStats";
import ServeStats from "@/components/ServeStats";
import type { StatusResponse, ResultsResponse } from "@/lib/api";

type AppState = "upload" | "processing" | "results" | "error";

export default function Home() {
  const [appState, setAppState] = useState<AppState>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUploadComplete = useCallback(async (newJobId: string) => {
    setJobId(newJobId);
    setAppState("processing");
    setError(null);

    try {
      const { pollUntilComplete } = await import("@/lib/api");
      const finalResults = await pollUntilComplete(
        newJobId,
        (statusUpdate) => setStatus(statusUpdate),
        2000
      );
      setResults(finalResults);
      setAppState("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Processing failed");
      setAppState("error");
    }
  }, []);

  const handleReset = useCallback(() => {
    setAppState("upload");
    setJobId(null);
    setStatus(null);
    setResults(null);
    setError(null);
  }, []);

  return (
    <main className="min-h-screen px-4 py-8 sm:px-8">
      {/* Header */}
      <header className="max-w-4xl mx-auto mb-12 text-center">
        <h1 className="text-4xl font-bold tracking-tight">
          🎾 Tennis Vision
        </h1>
        <p className="mt-2 text-[var(--color-muted)]">
          AI-powered tennis video analysis
        </p>
      </header>

      <div className="max-w-4xl mx-auto">
        {/* Upload state */}
        {appState === "upload" && (
          <VideoUpload onUploadComplete={handleUploadComplete} />
        )}

        {/* Processing state */}
        {appState === "processing" && status && (
          <div className="space-y-4">
            <ProcessingStatus status={status} />
            <p className="text-center text-xs text-[var(--color-muted)]">
              Job ID: {jobId}
            </p>
          </div>
        )}

        {/* Processing - waiting for first status */}
        {appState === "processing" && !status && (
          <div className="text-center space-y-4">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-[var(--color-accent)] mx-auto" />
            <p className="text-[var(--color-muted)]">Starting analysis...</p>
          </div>
        )}

        {/* Error state */}
        {appState === "error" && (
          <div className="max-w-xl mx-auto space-y-4">
            <div className="p-4 bg-red-900/30 border border-red-700 rounded-xl text-red-300">
              <h3 className="font-semibold mb-1">Analysis Failed</h3>
              <p className="text-sm">{error}</p>
              {jobId && (
                <p className="text-xs mt-2 text-red-400/70">Job ID: {jobId}</p>
              )}
            </div>
            <div className="text-center">
              <button
                onClick={handleReset}
                className="px-6 py-2 bg-[var(--color-card)] hover:bg-[var(--color-card-border)] border border-[var(--color-card-border)] rounded-lg transition-colors"
              >
                Try Another Video
              </button>
            </div>
          </div>
        )}

        {/* Results state */}
        {appState === "results" && results?.results && (
          <div className="space-y-8">
            {/* Video info */}
            <div className="text-center space-y-1">
              <h2 className="text-2xl font-bold">Analysis Complete</h2>
              <p className="text-[var(--color-muted)] text-sm">
                {results.results.video_filename} —{" "}
                {results.results.total_frames} frames @{" "}
                {results.results.fps.toFixed(1)} FPS
              </p>
            </div>

            {/* Court heatmap */}
            <CourtHeatmap
              ballPositions={results.results.ball_positions}
              title="Shot Placement Heatmap"
            />

            {/* Stats row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <RallyStats
                stats={results.results.rally_stats}
                rallies={results.results.rallies}
              />
              <ServeStats stats={results.results.serve_stats} />
            </div>

            {/* Visualization images */}
            {results.results.visualizations && (
              <VisualizationImages
                visualizations={results.results.visualizations}
              />
            )}

            {/* Reset button */}
            <div className="text-center pt-4">
              <button
                onClick={handleReset}
                className="px-6 py-2.5 bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white font-medium rounded-lg transition-colors"
              >
                Analyze Another Video
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

function VisualizationImages({
  visualizations,
}: {
  visualizations: {
    shot_heatmap: string | null;
    serve_heatmap_all: string | null;
    serve_heatmap_first: string | null;
    serve_heatmap_second: string | null;
  };
}) {
  const images = [
    { label: "Shot Heatmap", path: visualizations.shot_heatmap },
    { label: "All Serves", path: visualizations.serve_heatmap_all },
    { label: "1st Serves", path: visualizations.serve_heatmap_first },
    { label: "2nd Serves", path: visualizations.serve_heatmap_second },
  ].filter((img) => img.path);

  if (images.length === 0) return null;

  return (
    <div className="bg-[var(--color-card)] border border-[var(--color-card-border)] rounded-xl p-4 space-y-4">
      <h3 className="text-lg font-semibold">Generated Heatmaps</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {images.map((img) => (
          <div key={img.label} className="space-y-2">
            <p className="text-sm text-[var(--color-muted)]">{img.label}</p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`/api${img.path}`}
              alt={img.label}
              className="w-full rounded-lg border border-[var(--color-card-border)]"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
