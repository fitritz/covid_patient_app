import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ReferenceDot } from 'recharts';
import './VitalSignsTrends.css';

const VitalSignsTrends = ({ data }) => {
  const latestPoint = useMemo(() => (Array.isArray(data) && data.length > 0 ? data[data.length - 1] : null), [data]);

  const tooltipFormatter = (value, name) => {
    const units = {
      'Systolic BP': 'mmHg',
      'Diastolic BP': 'mmHg',
      'Heart Rate': 'bpm',
      'Body Temp (°C)': '°C',
    };
    return [`${value}${units[name] ? ' ' + units[name] : ''}`, name];
  };

  return (
    <div className="vital-signs-trends-card">
      <h3 className="vital-signs-title">Vital Signs Trends</h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={data}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <defs>
              <linearGradient id="gradSys" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#7c3aed" />
                <stop offset="100%" stopColor="#60a5fa" />
              </linearGradient>
              <linearGradient id="gradDia" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#22c55e" />
                <stop offset="100%" stopColor="#86efac" />
              </linearGradient>
              <linearGradient id="gradHR" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#f59e0b" />
                <stop offset="100%" stopColor="#f97316" />
              </linearGradient>
              <linearGradient id="gradTemp" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#ec4899" />
                <stop offset="100%" stopColor="#f43f5e" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip 
              formatter={tooltipFormatter} 
              cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.2)' }}
              contentStyle={{
                backgroundColor: 'rgba(18, 18, 18, 0.9)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#fff',
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
              }}
              itemStyle={{ color: '#fff' }}
              labelStyle={{ color: '#aaa', marginBottom: '5px' }}
            />
            <Legend wrapperStyle={{ paddingTop: 8, color: '#fff' }} />

            {/* Reference lines for common thresholds */}
            <ReferenceLine y={120} stroke="#7c3aed" strokeDasharray="6 6" ifOverflow="extendDomain" />
            <ReferenceLine y={80} stroke="#22c55e" strokeDasharray="6 6" ifOverflow="extendDomain" />
            <ReferenceLine y={100} stroke="#f59e0b" strokeDasharray="6 6" ifOverflow="extendDomain" />
            <ReferenceLine y={37.5} stroke="#ec4899" strokeDasharray="6 6" ifOverflow="extendDomain" />

            {/* Trend lines with gradients */}
            <Line type="monotone" dataKey="systolic_bp" stroke="url(#gradSys)" strokeWidth={2.5} dot={{ r: 2 }} activeDot={{ r: 5 }} name="Systolic BP" />
            <Line type="monotone" dataKey="diastolic_bp" stroke="url(#gradDia)" strokeWidth={2.5} dot={{ r: 2 }} activeDot={{ r: 5 }} name="Diastolic BP" />
            <Line type="monotone" dataKey="heart_rate" stroke="url(#gradHR)" strokeWidth={2.5} dot={{ r: 2 }} activeDot={{ r: 5 }} name="Heart Rate" />
            <Line type="monotone" dataKey="body_temp" stroke="url(#gradTemp)" strokeWidth={2.5} dot={{ r: 2 }} activeDot={{ r: 5 }} name="Body Temp (°C)" />

            {/* Highlight latest reading across series */}
            {latestPoint && (
              <>
                <ReferenceDot x={latestPoint.name} y={latestPoint.systolic_bp} r={4} fill="#7c3aed" />
                <ReferenceDot x={latestPoint.name} y={latestPoint.diastolic_bp} r={4} fill="#22c55e" />
                <ReferenceDot x={latestPoint.name} y={latestPoint.heart_rate} r={4} fill="#f59e0b" />
                <ReferenceDot x={latestPoint.name} y={latestPoint.body_temp} r={4} fill="#ec4899" />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default VitalSignsTrends;
