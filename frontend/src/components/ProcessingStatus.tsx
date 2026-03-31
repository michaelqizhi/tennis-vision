"use client";

import type { StatusResponse } from "@/lib/api";

interface ProcessingStatusProps {
  status: StatusResponse;
}

const STEP_LABELS: Record<string, string> = {
  ball_tracking: "Ball Tracking",
  court_detection: "Court Detection",
  rally_detection: "Rally Detection",
  rally_stats: "Rally Statistics",
  serve_analysis: "Serve Analysis",
  visualization: "Visualization",
};

export default function ProcessingStatus({ status }: ProcessingStatusProps) {
  const progressPct = Math.round(status.progress * 100);

  return (
    <div className="w-full max-w-xl mx-auto space-y-6">
      {/* Overall progress */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="font-medium">
            {status.status === "queued" ? "Queued..." : "Processing..."}
          </span>
          <span className="text-[var(--color-muted)]">{progressPct}%</span>
        </div>
        <div className="h-3 bg-[var(--color-card)] rounded-full overflow-hidden">
          <div
            className="h-full bg-[var(--color-accent)] rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        {status.current_step && (
          <p className="text-sm text-[var(--color-muted)]">
            {STEP_LABELS[status.current_step] || status.current_step}
          </p>
        )}
      </div>

      {/* Step-by-step progress */}
      {status.steps.length > 0 && (
        <div className="space-y-2">
          {status.steps.map((step) => (
            <div
              key={step.step}
              className="flex items-center gap-3 text-sm"
            >
              {/* Status icon */}
              <span className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
                {step.status === "complete" && (
                  <svg className="w-5 h-5 text-[var(--color-accent)]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
                {step.status === "processing" && (
                  <div className="w-4 h-4 rounded-full border-2 border-[var(--color-accent)] border-t-transparent animate-spin" />
                )}
                {step.status === "failed" && (
                  <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                )}
                {(step.status === "queued") && (
                  <div className="w-3 h-3 rounded-full bg-[var(--color-card-border)]" />
                )}
              </span>

              <span
                className={
                  step.status === "complete"
                    ? "text-[var(--color-accent)]"
                    : step.status === "processing"
                    ? "text-[var(--color-fg)]"
                    : step.status === "failed"
                    ? "text-red-400"
                    : "text-[var(--color-muted)]"
                }
              >
                {STEP_LABELS[step.step] || step.step}
              </span>
            </div>
          ))}
        </div>
      )}

      {status.error && (
        <div className="p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
          Error: {status.error}
        </div>
      )}
    </div>
  );
}
