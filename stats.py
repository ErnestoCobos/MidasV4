#!/usr/bin/env python3
"""
Script para mostrar rápidamente las estadísticas del bot sin tener que entrar en el menú interactivo.
Extrae información directamente del historial de operaciones y muestra un resumen completo.
"""

import os
import json
import argparse
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Directorio donde se guardan los gráficos
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

def get_trades_from_database(db_path="database/trading_data.db"):
    """Obtener historial de operaciones desde la base de datos"""
    if not os.path.exists(db_path):
        print(f"⚠️ Base de datos no encontrada en: {db_path}")
        return []
    
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Consultar todas las operaciones cerradas
        cursor.execute("""
        SELECT * FROM trades
        WHERE status = 'closed'
        ORDER BY entry_time
        """)
        
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return trades
    except Exception as e:
        print(f"❌ Error al leer la base de datos: {str(e)}")
        return []

def calculate_metrics(trades):
    """Calcular métricas de rendimiento"""
    if not trades:
        return {
            "total_trades": 0,
            "profitable_trades": 0,
            "win_rate": 0,
            "total_profit_loss": 0,
            "avg_profit": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "best_symbols": [],
            "worst_symbols": []
        }
    
    total_trades = len(trades)
    profitable_trades = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
    total_profit_loss = sum(t.get('profit_loss', 0) for t in trades)
    
    # Calcular métricas básicas
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit = sum(t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) > 0) / max(1, profitable_trades)
    avg_loss = sum(abs(t.get('profit_loss', 0)) for t in trades if t.get('profit_loss', 0) < 0) / max(1, total_trades - profitable_trades)
    
    # Calcular profit factor
    profit_factor = avg_profit / avg_loss if avg_loss > 0 else float('inf') if avg_profit > 0 else 0
    
    # Rendimiento por símbolo
    symbol_performance = {}
    for trade in trades:
        symbol = trade.get('symbol', 'desconocido')
        if symbol not in symbol_performance:
            symbol_performance[symbol] = {
                'trades': 0,
                'profit_loss': 0,
                'wins': 0
            }
        
        symbol_performance[symbol]['trades'] += 1
        symbol_performance[symbol]['profit_loss'] += trade.get('profit_loss', 0)
        if trade.get('profit_loss', 0) > 0:
            symbol_performance[symbol]['wins'] += 1
    
    # Ordenar por rendimiento
    best_symbols = sorted(
        symbol_performance.items(),
        key=lambda x: x[1]['profit_loss'],
        reverse=True
    )[:3]
    
    worst_symbols = sorted(
        symbol_performance.items(),
        key=lambda x: x[1]['profit_loss']
    )[:3]
    
    return {
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "win_rate": win_rate,
        "total_profit_loss": total_profit_loss,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "best_symbols": [{'symbol': s, 'profit': p['profit_loss'], 'trades': p['trades']} for s, p in best_symbols],
        "worst_symbols": [{'symbol': s, 'profit': p['profit_loss'], 'trades': p['trades']} for s, p in worst_symbols]
    }

def plot_performance_chart(trades, output_file=None):
    """Generar gráfico de rendimiento acumulado"""
    if not trades:
        return None
    
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([{
            'time': datetime.fromisoformat(trade['exit_time']) if isinstance(trade['exit_time'], str) else trade['exit_time'],
            'profit_loss': trade['profit_loss'],
            'symbol': trade['symbol']
        } for trade in trades if trade.get('exit_time')])
        
        if df.empty:
            return None
        
        # Ordenar por tiempo
        df = df.sort_values('time')
        
        # Calcular P/L acumulado
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        
        # Crear gráfico
        plt.figure(figsize=(12, 6))
        
        # Graficar P/L acumulado
        plt.plot(df['time'], df['cumulative_pnl'], label='P/L Acumulado', color='blue')
        
        # Marcar operaciones individuales
        colors = ['green' if pl >= 0 else 'red' for pl in df['profit_loss']]
        plt.scatter(df['time'], df['cumulative_pnl'], c=colors, s=30)
        
        # Agregar línea horizontal en y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Establecer etiquetas y título
        plt.title('Rendimiento de Trading')
        plt.xlabel('Tiempo')
        plt.ylabel('Ganancia/Pérdida Acumulada')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar en archivo
        if not output_file:
            output_file = f"{CHARTS_DIR}/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    except Exception as e:
        print(f"❌ Error al generar gráfico: {str(e)}")
        return None

def display_trade_summary(trades):
    """Mostrar un resumen de las operaciones recientes"""
    if not trades:
        print("No hay operaciones registradas.")
        return
    
    # Mostrar las últimas 10 operaciones
    recent_trades = trades[-10:]
    recent_trades.reverse()  # Ordenar de más reciente a más antigua
    
    # Preparar tabla para mostrar
    table_data = []
    for trade in recent_trades:
        profit = trade.get('profit_loss', 0)
        profit_str = f"+{profit:.8f}" if profit >= 0 else f"{profit:.8f}"
        
        # Convertir timestamps a formato legible
        entry_time = trade.get('entry_time')
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time).strftime('%Y-%m-%d %H:%M')
            except ValueError:
                pass
        
        exit_time = trade.get('exit_time')
        if isinstance(exit_time, str):
            try:
                exit_time = datetime.fromisoformat(exit_time).strftime('%Y-%m-%d %H:%M')
            except ValueError:
                pass
        
        table_data.append([
            trade.get('symbol', 'N/A'),
            trade.get('side', 'N/A'),
            entry_time,
            exit_time,
            profit_str,
            trade.get('exit_reason', 'N/A')
        ])
    
    headers = ["Símbolo", "Dirección", "Entrada", "Salida", "P/L", "Razón"]
    print("\n📜 ÚLTIMAS OPERACIONES:")
    print(tabulate(table_data, headers=headers, tablefmt="simple"))

def display_performance_summary(metrics):
    """Mostrar un resumen de rendimiento"""
    print("\n📊 RESUMEN DE RENDIMIENTO:")
    print(f"Total de operaciones: {metrics['total_trades']}")
    print(f"Operaciones rentables: {metrics['profitable_trades']} ({metrics['win_rate']:.2f}%)")
    print(f"Ganancia/Pérdida total: {metrics['total_profit_loss']:.8f}")
    
    if metrics['total_trades'] > 0:
        print(f"Ganancia media: {metrics['avg_profit']:.8f}")
        print(f"Pérdida media: {metrics['avg_loss']:.8f}")
        print(f"Factor de beneficio: {metrics['profit_factor']:.2f}")
    
    if metrics['best_symbols']:
        print("\n🔝 MEJORES PARES:")
        for i, symbol_data in enumerate(metrics['best_symbols'], 1):
            symbol = symbol_data['symbol']
            profit = symbol_data['profit']
            trades = symbol_data['trades']
            print(f"{i}. {symbol}: {profit:.8f} ({trades} operaciones)")
    
    if metrics['worst_symbols']:
        print("\n⚠️ PEORES PARES:")
        for i, symbol_data in enumerate(metrics['worst_symbols'], 1):
            symbol = symbol_data['symbol']
            profit = symbol_data['profit']
            trades = symbol_data['trades']
            print(f"{i}. {symbol}: {profit:.8f} ({trades} operaciones)")

def main():
    parser = argparse.ArgumentParser(description="Mostrar estadísticas de rendimiento del bot")
    parser.add_argument("--db", type=str, help="Ruta de la base de datos SQLite", 
                      default="database/trading_data.db")
    parser.add_argument("--chart", action="store_true", help="Generar gráfico de rendimiento")
    parser.add_argument("--output", type=str, help="Archivo de salida para el gráfico",
                      default=None)
    parser.add_argument("--full", action="store_true", help="Mostrar todas las estadísticas")
    
    args = parser.parse_args()
    
    # Buscar base de datos en directorio actual o raíz del proyecto
    db_path = args.db
    if not os.path.exists(db_path):
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), db_path)
        if os.path.exists(alt_path):
            db_path = alt_path
    
    # Obtener historial de operaciones
    trades = get_trades_from_database(db_path)
    
    if not trades:
        print("⚠️ No se encontraron operaciones en la base de datos.")
        return
    
    # Calcular métricas
    metrics = calculate_metrics(trades)
    
    # Mostrar resultados
    display_performance_summary(metrics)
    
    if args.full:
        display_trade_summary(trades)
    
    # Generar gráfico si se solicita
    if args.chart:
        chart_file = plot_performance_chart(trades, args.output)
        if chart_file:
            print(f"\n✅ Gráfico de rendimiento guardado en: {chart_file}")

if __name__ == "__main__":
    print("🚀 MidasScalpingBot v4 - Estadísticas de Rendimiento")
    main()