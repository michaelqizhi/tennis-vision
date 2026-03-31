"use client";

import { useState, useCallback } from "react";
import type { BallPosition } from "@/lib/api";

interface CourtHeatmapProps {
  ballPositions: BallPosition[];
  title?: string;
  filter?: "all" | "serves" | "groundstrokes";
}

// ITF court dimensions in meters
const COURT_LENGTH = 23.77;
const COURT_WIDTH = 10.97;
const SINGLES_WIDTH = 8.23;
const SERVICE_LINE = 6.40; // from net
const NET_Y = COURT_LENGTH / 2;

// SVG viewBox padding
const PAD = 1.5;
const VB_W = COURT_WIDTH + PAD * 2;
const VB_H = COURT_LENGTH + PAD * 2;

interface TooltipData {
  x: number;
  y: number;
  courtX: number;
  courtY: number;
  frame: number;
}

export default function CourtHeatmap({
  ballPositions,
  title = "Shot Placement Heatmap",
}: CourtHeatmapProps) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  // Filter positions that have valid court coordinates
  const validPositions = ballPositions.filter(
    (p) => p.court_x != null && p.court_y != null
  );

  const handleHover = useCallback(
    (pos: BallPosition, svgX: number, svgY: number) => {
      setTooltip({
        x: svgX,
        y: svgY,
        courtX: pos.court_x!,
        courtY: pos.court_y!,
        frame: pos.frame_number,
      });
    },
    []
  );

  // Compute density for color intensity
  const maxDensity = computeMaxDensity(validPositions);

  return (
    <div className="bg-[var(--color-card)] border border-[var(--color-card-border)] rounded-xl p-4">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>

      {validPositions.length === 0 ? (
        <p className="text-[var(--color-muted)] text-sm text-center py-8">
          No court-mapped ball positions available.
        </p>
      ) : (
        <div className="relative">
          <svg
            viewBox={`${-PAD} ${-PAD} ${VB_W} ${VB_H}`}
            className="w-full max-w-md mx-auto"
            style={{ aspectRatio: `${VB_W} / ${VB_H}` }}
            onMouseLeave={() => setTooltip(null)}
          >
            {/* Court surface */}
            <rect
              x={0} y={0}
              width={COURT_WIDTH} height={COURT_LENGTH}
              fill="#1a5f2a" stroke="#fff" strokeWidth={0.05}
            />

            {/* Singles sidelines */}
            {(() => {
              const singleOffset = (COURT_WIDTH - SINGLES_WIDTH) / 2;
              return (
                <>
                  <line x1={singleOffset} y1={0} x2={singleOffset} y2={COURT_LENGTH}
                    stroke="#fff" strokeWidth={0.05} />
                  <line x1={COURT_WIDTH - singleOffset} y1={0} x2={COURT_WIDTH - singleOffset} y2={COURT_LENGTH}
                    stroke="#fff" strokeWidth={0.05} />
                </>
              );
            })()}

            {/* Net */}
            <line x1={0} y1={NET_Y} x2={COURT_WIDTH} y2={NET_Y}
              stroke="#fff" strokeWidth={0.08} />

            {/* Service lines */}
            {(() => {
              const singleOffset = (COURT_WIDTH - SINGLES_WIDTH) / 2;
              return (
                <>
                  <line x1={singleOffset} y1={NET_Y - SERVICE_LINE}
                    x2={COURT_WIDTH - singleOffset} y2={NET_Y - SERVICE_LINE}
                    stroke="#fff" strokeWidth={0.05} />
                  <line x1={singleOffset} y1={NET_Y + SERVICE_LINE}
                    x2={COURT_WIDTH - singleOffset} y2={NET_Y + SERVICE_LINE}
                    stroke="#fff" strokeWidth={0.05} />
                </>
              );
            })()}

            {/* Center service line */}
            <line x1={COURT_WIDTH / 2} y1={NET_Y - SERVICE_LINE}
              x2={COURT_WIDTH / 2} y2={NET_Y + SERVICE_LINE}
              stroke="#fff" strokeWidth={0.05} />

            {/* Center marks */}
            <line x1={COURT_WIDTH / 2} y1={0} x2={COURT_WIDTH / 2} y2={0.3}
              stroke="#fff" strokeWidth={0.05} />
            <line x1={COURT_WIDTH / 2} y1={COURT_LENGTH} x2={COURT_WIDTH / 2} y2={COURT_LENGTH - 0.3}
              stroke="#fff" strokeWidth={0.05} />

            {/* Ball positions as heatmap dots */}
            {validPositions.map((pos, i) => {
              // Backend uses center-origin coords; SVG uses top-left origin
              const cx = pos.court_x! + COURT_WIDTH / 2;
              const cy = pos.court_y! + COURT_LENGTH / 2;
              // Skip positions outside the court area (with margin)
              if (cx < -1 || cx > COURT_WIDTH + 1 || cy < -1 || cy > COURT_LENGTH + 1) return null;
              const density = computeLocalDensity(pos, validPositions);
              const opacity = 0.3 + 0.7 * (density / Math.max(maxDensity, 1));
              return (
                <circle
                  key={i}
                  cx={cx}
                  cy={cy}
                  r={0.15}
                  fill={`rgba(34, 197, 94, ${opacity})`}
                  stroke="none"
                  className="cursor-pointer hover:r-[0.25]"
                  onMouseEnter={() => handleHover(pos, cx, cy)}
                />
              );
            })}
          </svg>

          {/* Tooltip */}
          {tooltip && (
            <div
              className="absolute bg-black/90 text-white text-xs px-2 py-1 rounded pointer-events-none z-10 whitespace-nowrap"
              style={{
                left: `${((tooltip.x + PAD) / VB_W) * 100}%`,
                top: `${((tooltip.y + PAD) / VB_H) * 100}%`,
                transform: "translate(-50%, -120%)",
              }}
            >
              Frame {tooltip.frame} — ({tooltip.courtX.toFixed(2)}m, {tooltip.courtY.toFixed(2)}m)
            </div>
          )}

          <p className="text-center text-xs text-[var(--color-muted)] mt-2">
            {validPositions.length} positions mapped
          </p>
        </div>
      )}
    </div>
  );
}

/** Count how many other positions are within 0.5m radius */
function computeLocalDensity(pos: BallPosition, all: BallPosition[]): number {
  const R = 0.5;
  // Use transformed (SVG) coordinates for density
  const px = (pos.court_x ?? 0) + COURT_WIDTH / 2;
  const py = (pos.court_y ?? 0) + COURT_LENGTH / 2;
  let count = 0;
  for (const other of all) {
    const ox = (other.court_x ?? 0) + COURT_WIDTH / 2;
    const oy = (other.court_y ?? 0) + COURT_LENGTH / 2;
    const dx = px - ox;
    const dy = py - oy;
    if (dx * dx + dy * dy <= R * R) count++;
  }
  return count;
}

function computeMaxDensity(positions: BallPosition[]): number {
  if (positions.length === 0) return 0;
  // Sample to avoid O(n²) on large datasets
  const sample = positions.length > 200
    ? positions.filter((_, i) => i % Math.ceil(positions.length / 200) === 0)
    : positions;
  let max = 0;
  for (const pos of sample) {
    max = Math.max(max, computeLocalDensity(pos, positions));
  }
  return max;
}
