"use client";

import { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Clock,
  RefreshCw,
  AlertCircle,
  BarChart2,
  Layers,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface BotStatus {
  is_running: boolean;
  last_run: string;
  scan_interval: number;
  current_regime: string;
  session: string;
  account_balance: number;
  daily_pnl: number;
  win_rate: number;
  open_positions_count: number;
  error?: string;
}

interface Position {
  ticket: number;
  symbol: string;
  direction: "BUY" | "SELL";
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  current_profit: number;
  volume: number;
  open_time: string;
}

interface Trade {
  ticket: number;
  symbol: string;
  direction: "BUY" | "SELL";
  entry_price: number;
  exit_price: number;
  profit: number;
  volume: number;
  open_time: string;
  close_time: string;
  duration: string;
}

interface EquityPoint {
  time: string;
  equity: number;
  balance: number;
}

interface ApiData {
  status: BotStatus | null;
  positions: Position[];
  trades: Trade[];
  equityCurve: EquityPoint[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtUSD(n: number): string {
  return `$${fmt(Math.abs(n))}`;
}

function pnlColor(val: number): string {
  return val >= 0 ? "text-emerald-400" : "text-red-400";
}

function pnlSign(val: number): string {
  return val >= 0 ? "+" : "-";
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function StatCard({
  icon,
  label,
  value,
  sub,
  accent = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div className="stat-card">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          {label}
        </span>
        <span className={accent ? "text-yellow-400" : "text-gray-500"}>
          {icon}
        </span>
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs mt-1">{sub}</div>}
    </div>
  );
}

function DirectionBadge({ direction }: { direction: "BUY" | "SELL" }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${
        direction === "BUY"
          ? "bg-emerald-900/60 text-emerald-400 border border-emerald-700"
          : "bg-red-900/60 text-red-400 border border-red-700"
      }`}
    >
      {direction === "BUY" ? (
        <TrendingUp size={10} className="mr-1" />
      ) : (
        <TrendingDown size={10} className="mr-1" />
      )}
      {direction}
    </span>
  );
}

function SectionHeader({
  icon,
  title,
  badge,
}: {
  icon: React.ReactNode;
  title: string;
  badge?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-yellow-400">{icon}</span>
      <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-wider">
        {title}
      </h2>
      {badge}
    </div>
  );
}

function EmptyRow({ cols, message }: { cols: number; message: string }) {
  return (
    <tr>
      <td
        colSpan={cols}
        className="px-4 py-8 text-center text-gray-500 text-sm"
      >
        {message}
      </td>
    </tr>
  );
}

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-gray-400 mb-2">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }} className="font-semibold">
          {p.name}: ${fmt(p.value)}
        </p>
      ))}
    </div>
  );
};

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [data, setData] = useState<ApiData>({
    status: null,
    positions: [],
    trades: [],
    equityCurve: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAll = useCallback(async () => {
    setRefreshing(true);
    try {
      const [statusRes, posRes, tradesRes, equityRes] = await Promise.allSettled([
        fetch("/api/status"),
        fetch("/api/positions"),
        fetch("/api/trades"),
        fetch("/api/equity-curve"),
      ]);

      const parse = async (r: PromiseSettledResult<Response>) => {
        if (r.status === "fulfilled" && r.value.ok) {
          return r.value.json().catch(() => null);
        }
        return null;
      };

      const [status, positions, trades, equityCurve] = await Promise.all([
        parse(statusRes),
        parse(posRes),
        parse(tradesRes),
        parse(equityRes),
      ]);

      setData({
        status: status ?? null,
        positions: positions?.positions ?? [],
        trades: trades?.trades ?? [],
        equityCurve: equityCurve?.equity_curve ?? [],
      });
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch data");
    } finally {
      setLoading(false);
      setRefreshing(false);
      setLastRefresh(new Date());
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 30_000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  const { status, positions, trades, equityCurve } = data;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-900">
      {/* ── Header ── */}
      <header className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur border-b border-gray-800 px-6 py-4">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-yellow-400/10 border border-yellow-400/30 flex items-center justify-center">
              <span className="text-yellow-400 font-bold text-sm">G</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-none">
                GOLDAI Trading Bot
              </h1>
              <p className="text-xs text-gray-500 mt-0.5">
                XAUUSD AI-Powered Trading System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {status && (
              <div className="flex items-center gap-2">
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    status.is_running
                      ? "bg-emerald-400 status-dot-active"
                      : "bg-red-400"
                  }`}
                />
                <span className="text-xs text-gray-400">
                  {status.is_running ? "Bot Running" : "Bot Stopped"}
                </span>
              </div>
            )}

            <button
              onClick={fetchAll}
              disabled={refreshing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-600 transition-colors text-xs disabled:opacity-50"
            >
              <RefreshCw
                size={12}
                className={refreshing ? "animate-spin" : ""}
              />
              Refresh
            </button>

            {lastRefresh && (
              <span className="text-xs text-gray-600 hidden sm:block">
                Updated {lastRefresh.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto px-6 py-6 space-y-6">
        {/* ── Error Banner ── */}
        {error && (
          <div className="flex items-center gap-3 bg-red-900/30 border border-red-700/50 rounded-xl px-4 py-3 text-red-400 text-sm">
            <AlertCircle size={16} className="shrink-0" />
            <span>
              API unavailable — showing last known data. ({error})
            </span>
          </div>
        )}

        {/* ── Loading ── */}
        {loading && !error && (
          <div className="flex items-center justify-center py-20">
            <div className="flex flex-col items-center gap-3">
              <RefreshCw size={24} className="text-yellow-400 animate-spin" />
              <span className="text-gray-400 text-sm">
                Loading dashboard...
              </span>
            </div>
          </div>
        )}

        {/* ── Stats Row ── */}
        {!loading && (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                icon={<DollarSign size={16} />}
                label="Account Balance"
                value={
                  status ? `$${fmt(status.account_balance)}` : "—"
                }
                accent
              />
              <StatCard
                icon={
                  status && status.daily_pnl >= 0 ? (
                    <TrendingUp size={16} />
                  ) : (
                    <TrendingDown size={16} />
                  )
                }
                label="Daily P&L"
                value={
                  status
                    ? `${pnlSign(status.daily_pnl)}${fmtUSD(status.daily_pnl)}`
                    : "—"
                }
                sub={
                  status && (
                    <span className={pnlColor(status.daily_pnl)}>
                      {status.daily_pnl >= 0 ? "Profitable day" : "Losing day"}
                    </span>
                  )
                }
              />
              <StatCard
                icon={<Target size={16} />}
                label="Win Rate"
                value={status ? `${fmt(status.win_rate, 1)}%` : "—"}
                sub={
                  <span className="text-gray-500">
                    Based on closed trades
                  </span>
                }
              />
              <StatCard
                icon={<Layers size={16} />}
                label="Open Positions"
                value={
                  status
                    ? String(status.open_positions_count)
                    : String(positions.length)
                }
                sub={
                  <span className="text-gray-500">Active trades</span>
                }
              />
            </div>

            {/* ── Main Grid ── */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              {/* ── Equity Curve (spans 2 cols) ── */}
              <div className="xl:col-span-2 card">
                <SectionHeader
                  icon={<BarChart2 size={16} />}
                  title="Equity Curve"
                />
                {equityCurve.length === 0 ? (
                  <div className="flex items-center justify-center h-56 text-gray-600 text-sm">
                    No equity data available
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={260}>
                    <AreaChart
                      data={equityCurve}
                      margin={{ top: 4, right: 4, left: 0, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient
                          id="equityGradient"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="5%"
                            stopColor="#FFD700"
                            stopOpacity={0.25}
                          />
                          <stop
                            offset="95%"
                            stopColor="#FFD700"
                            stopOpacity={0}
                          />
                        </linearGradient>
                        <linearGradient
                          id="balanceGradient"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="5%"
                            stopColor="#6366f1"
                            stopOpacity={0.2}
                          />
                          <stop
                            offset="95%"
                            stopColor="#6366f1"
                            stopOpacity={0}
                          />
                        </linearGradient>
                      </defs>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#374151"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="time"
                        tick={{ fontSize: 10, fill: "#6b7280" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: "#6b7280" }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                        width={50}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Area
                        type="monotone"
                        dataKey="balance"
                        stroke="#6366f1"
                        strokeWidth={1.5}
                        fill="url(#balanceGradient)"
                        name="Balance"
                        dot={false}
                      />
                      <Area
                        type="monotone"
                        dataKey="equity"
                        stroke="#FFD700"
                        strokeWidth={2}
                        fill="url(#equityGradient)"
                        name="Equity"
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                )}
              </div>

              {/* ── Bot Status Card ── */}
              <div className="card">
                <SectionHeader
                  icon={<Activity size={16} />}
                  title="Bot Status"
                />
                {status ? (
                  <dl className="space-y-3">
                    {[
                      {
                        label: "Status",
                        value: (
                          <span
                            className={
                              status.is_running
                                ? "text-emerald-400"
                                : "text-red-400"
                            }
                          >
                            {status.is_running ? "● Running" : "● Stopped"}
                          </span>
                        ),
                      },
                      {
                        label: "Last Run",
                        value: status.last_run
                          ? new Date(status.last_run).toLocaleTimeString()
                          : "—",
                      },
                      {
                        label: "Scan Interval",
                        value: `${status.scan_interval}s`,
                      },
                      {
                        label: "Regime",
                        value: (
                          <span className="px-2 py-0.5 rounded bg-yellow-400/10 text-yellow-400 text-xs border border-yellow-400/20">
                            {status.current_regime || "—"}
                          </span>
                        ),
                      },
                      {
                        label: "Session",
                        value: (
                          <span className="px-2 py-0.5 rounded bg-indigo-400/10 text-indigo-400 text-xs border border-indigo-400/20">
                            {status.session || "—"}
                          </span>
                        ),
                      },
                    ].map(({ label, value }) => (
                      <div
                        key={label}
                        className="flex items-center justify-between py-1 border-b border-gray-700/50 last:border-0"
                      >
                        <dt className="text-xs text-gray-500">{label}</dt>
                        <dd className="text-xs font-medium text-gray-200">
                          {value}
                        </dd>
                      </div>
                    ))}
                  </dl>
                ) : (
                  <div className="flex items-center justify-center h-40 text-gray-600 text-sm">
                    No status data
                  </div>
                )}

                {/* Refresh interval indicator */}
                <div className="mt-4 pt-3 border-t border-gray-700/50 flex items-center gap-1.5 text-xs text-gray-600">
                  <Clock size={10} />
                  Auto-refreshes every 30s
                </div>
              </div>
            </div>

            {/* ── Open Positions Table ── */}
            <div className="card overflow-hidden">
              <SectionHeader
                icon={<Layers size={16} />}
                title="Open Positions"
                badge={
                  positions.length > 0 ? (
                    <span className="px-2 py-0.5 rounded-full bg-yellow-400/10 text-yellow-400 text-xs border border-yellow-400/20">
                      {positions.length}
                    </span>
                  ) : undefined
                }
              />
              <div className="overflow-x-auto -mx-4 -mb-4">
                <table className="min-w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left">Ticket</th>
                      <th className="text-left">Symbol</th>
                      <th className="text-left">Dir</th>
                      <th className="text-right">Entry</th>
                      <th className="text-right">Stop Loss</th>
                      <th className="text-right">Take Profit</th>
                      <th className="text-right">Volume</th>
                      <th className="text-right">P&L</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700/50">
                    {positions.length === 0 ? (
                      <EmptyRow
                        cols={8}
                        message="No open positions"
                      />
                    ) : (
                      positions.map((pos) => (
                        <tr
                          key={pos.ticket}
                          className="hover:bg-gray-700/30 transition-colors"
                        >
                          <td className="text-xs text-gray-400 font-mono">
                            #{pos.ticket}
                          </td>
                          <td className="text-sm font-semibold text-white">
                            {pos.symbol}
                          </td>
                          <td>
                            <DirectionBadge direction={pos.direction} />
                          </td>
                          <td className="text-right text-sm text-gray-200 font-mono">
                            {fmt(pos.entry_price, 2)}
                          </td>
                          <td className="text-right text-sm text-red-400 font-mono">
                            {fmt(pos.stop_loss, 2)}
                          </td>
                          <td className="text-right text-sm text-emerald-400 font-mono">
                            {fmt(pos.take_profit, 2)}
                          </td>
                          <td className="text-right text-sm text-gray-300 font-mono">
                            {pos.volume}
                          </td>
                          <td
                            className={`text-right text-sm font-semibold font-mono ${pnlColor(
                              pos.current_profit
                            )}`}
                          >
                            {pnlSign(pos.current_profit)}
                            {fmtUSD(pos.current_profit)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* ── Recent Trades Table ── */}
            <div className="card overflow-hidden">
              <SectionHeader
                icon={<Clock size={16} />}
                title="Recent Trades"
                badge={
                  trades.length > 0 ? (
                    <span className="px-2 py-0.5 rounded-full bg-gray-700 text-gray-400 text-xs">
                      Last {Math.min(trades.length, 10)}
                    </span>
                  ) : undefined
                }
              />
              <div className="overflow-x-auto -mx-4 -mb-4">
                <table className="min-w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left">Ticket</th>
                      <th className="text-left">Symbol</th>
                      <th className="text-left">Dir</th>
                      <th className="text-right">Entry</th>
                      <th className="text-right">Exit</th>
                      <th className="text-right">Volume</th>
                      <th className="text-right">Duration</th>
                      <th className="text-right">Profit</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-700/50">
                    {trades.length === 0 ? (
                      <EmptyRow
                        cols={8}
                        message="No recent trades"
                      />
                    ) : (
                      trades.slice(0, 10).map((trade) => (
                        <tr
                          key={trade.ticket}
                          className="hover:bg-gray-700/30 transition-colors"
                        >
                          <td className="text-xs text-gray-400 font-mono">
                            #{trade.ticket}
                          </td>
                          <td className="text-sm font-semibold text-white">
                            {trade.symbol}
                          </td>
                          <td>
                            <DirectionBadge direction={trade.direction} />
                          </td>
                          <td className="text-right text-sm text-gray-200 font-mono">
                            {fmt(trade.entry_price, 2)}
                          </td>
                          <td className="text-right text-sm text-gray-200 font-mono">
                            {fmt(trade.exit_price, 2)}
                          </td>
                          <td className="text-right text-sm text-gray-300 font-mono">
                            {trade.volume}
                          </td>
                          <td className="text-right text-xs text-gray-500">
                            {trade.duration}
                          </td>
                          <td
                            className={`text-right text-sm font-semibold font-mono ${pnlColor(
                              trade.profit
                            )}`}
                          >
                            {pnlSign(trade.profit)}
                            {fmtUSD(trade.profit)}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="mt-10 border-t border-gray-800 px-6 py-4 text-center text-xs text-gray-700">
        GOLDAI Dashboard · XAUUSD Trading Bot ·{" "}
        <span className="text-yellow-600">Not financial advice</span>
      </footer>
    </div>
  );
}
