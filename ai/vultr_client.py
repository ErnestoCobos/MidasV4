import requests
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger('VultrAI')

class VultrInferenceClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-instruct-fp8"):
        self.api_key = api_key
        self.base_url = "https://api.vultrinference.com/v1"
        self.default_model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_market_conditions(self, 
                                 market_data: Dict[str, Any], 
                                 performance_history: Dict[str, Any],
                                 current_parameters: Dict[str, Any],
                                 model: str = None) -> Dict[str, Any]:
        """
        Analiza condiciones de mercado y recomienda ajustes a parámetros
        
        Args:
            market_data: Indicadores técnicos actuales
            performance_history: Historial de rendimiento del bot
            current_parameters: Parámetros actuales de la estrategia
            model: Modelo a usar (opcional)
            
        Returns:
            Diccionario con recomendaciones de parámetros
        """
        try:
            # Preparar el prompt para el modelo
            prompt = self._build_analysis_prompt(
                market_data, 
                performance_history, 
                current_parameters
            )
            
            # Usar modelo especificado o el predeterminado
            model_to_use = model or self.default_model
            
            # Configurar solicitud a la API
            messages = [
                {"role": "system", "content": "Eres un asistente experto en trading algorítmico especializado en análisis técnico y optimización de estrategias de scalping para criptomonedas."},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": model_to_use,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.2  # Baja temperatura para respuestas más deterministas
            }
            
            # Realizar solicitud a la API
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                result = response.json()
                # Extraer y procesar sugerencias
                suggestions = self._extract_suggestions(
                    result["choices"][0]["message"]["content"]
                )
                return suggestions
            else:
                logger.error(f"Error en respuesta de Vultr: {response.status_code}")
                return {"error": f"Error API: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error en análisis de mercado: {str(e)}")
            return {"error": str(e)}
    
    def generate_trading_signals(self, 
                               market_data: pd.DataFrame, 
                               symbol: str,
                               timeframe: str,
                               model: str = None) -> Dict[str, Any]:
        """
        Genera señales de trading basadas en análisis de datos de mercado usando LLM
        
        Args:
            market_data: DataFrame con datos OHLCV e indicadores técnicos
            symbol: Par de trading (ej. BTCUSDT)
            timeframe: Marco temporal (ej. 1m, 5m, 1h)
            model: Modelo a usar (opcional)
            
        Returns:
            Diccionario con señales y análisis de trading
        """
        try:
            # Preparar los datos para el prompt
            # Usar solo las últimas N velas para no exceder contexto
            recent_data = market_data.tail(30).copy()
            
            # Calcular algunos indicadores básicos si no existen
            if 'rsi' not in recent_data.columns:
                recent_data['rsi'] = self._calculate_rsi(recent_data['close'])
            if 'ema_9' not in recent_data.columns:
                recent_data['ema_9'] = recent_data['close'].ewm(span=9).mean()
            if 'ema_21' not in recent_data.columns:
                recent_data['ema_21'] = recent_data['close'].ewm(span=21).mean()
            
            # Formatear datos para el prompt
            data_str = recent_data.to_string()
            
            # Construir el prompt para generación de señales
            prompt = self._build_signal_prompt(data_str, symbol, timeframe)
            
            # Usar modelo especificado o el predeterminado
            model_to_use = model or self.default_model
            
            # Configurar solicitud a la API
            messages = [
                {"role": "system", "content": "Eres un asistente experto en trading algorítmico con amplia experiencia en análisis técnico y estrategias de scalping. Tu objetivo es identificar oportunidades de trading de alta probabilidad basadas en análisis de datos de mercado."},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": model_to_use,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.1  # Temperatura muy baja para respuestas consistentes
            }
            
            # Realizar solicitud a la API
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                result = response.json()
                # Extraer y procesar señales
                signals = self._extract_signals(
                    result["choices"][0]["message"]["content"]
                )
                # Añadir metadatos
                signals["symbol"] = symbol
                signals["timeframe"] = timeframe
                signals["timestamp"] = pd.Timestamp.now().isoformat()
                
                return signals
            else:
                logger.error(f"Error en respuesta de Vultr: {response.status_code}")
                return {"error": f"Error API: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error generando señales: {str(e)}")
            return {"error": str(e)}
    
    def predict_price_movement(self, 
                             market_data: pd.DataFrame, 
                             symbol: str,
                             timeframe: str,
                             horizon: str = "1h",
                             model: str = None) -> Dict[str, Any]:
        """
        Predice el movimiento de precio a corto plazo
        
        Args:
            market_data: DataFrame con datos OHLCV e indicadores técnicos
            symbol: Par de trading (ej. BTCUSDT)
            timeframe: Marco temporal (ej. 1m, 5m, 1h)
            horizon: Horizonte de predicción (ej. 1h, 4h, 1d)
            model: Modelo a usar (opcional)
            
        Returns:
            Diccionario con predicción de movimiento de precio
        """
        try:
            # Preparar los datos para el prompt
            recent_data = market_data.tail(20).copy()
            price_data_str = recent_data[['open', 'high', 'low', 'close', 'volume']].to_string()
            
            # Obtener algunas estadísticas de mercado
            current_price = market_data['close'].iloc[-1]
            daily_high = market_data['high'].max()
            daily_low = market_data['low'].min()
            avg_volume = market_data['volume'].mean()
            
            # Construir el prompt para predicción
            prompt = f"""
Analiza los siguientes datos de mercado para {symbol} en timeframe {timeframe} y predice el movimiento de precio en las próximas {horizon}.

## DATOS RECIENTES (últimas 20 velas)
{price_data_str}

## ESTADÍSTICAS DE MERCADO
- Precio actual: {current_price}
- Máximo reciente: {daily_high}
- Mínimo reciente: {daily_low}
- Volumen promedio: {avg_volume}

Por favor, proporciona tu predicción SOLO en formato JSON con la siguiente estructura exacta:
```json
{{
  "direction": "up|down|sideways",
  "price_target": float,
  "confidence": float,  # entre 0.0 y 1.0
  "analysis": "Breve explicación de tu predicción basada en los datos"
}}
```
"""
            # Usar modelo especificado o el predeterminado
            model_to_use = model or self.default_model
            
            # Configurar solicitud a la API
            messages = [
                {"role": "system", "content": "Eres un experto en análisis técnico y predicción de precios para mercados financieros, especialmente criptomonedas."},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": model_to_use,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.2
            }
            
            # Realizar solicitud a la API
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                result = response.json()
                prediction = self._extract_json(result["choices"][0]["message"]["content"])
                prediction["symbol"] = symbol
                prediction["timeframe"] = timeframe
                prediction["horizon"] = horizon
                prediction["current_price"] = current_price
                prediction["timestamp"] = pd.Timestamp.now().isoformat()
                
                return prediction
            else:
                logger.error(f"Error en respuesta de Vultr: {response.status_code}")
                return {"error": f"Error API: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error prediciendo precio: {str(e)}")
            return {"error": str(e)}
    
    def _build_analysis_prompt(self, 
                              market_data: Dict[str, Any],
                              performance_history: Dict[str, Any],
                              current_parameters: Dict[str, Any]) -> str:
        """Construye prompt para análisis de mercado"""
        # Formatear datos para el prompt
        market_summary = json.dumps(market_data, indent=2)
        performance_summary = json.dumps(performance_history, indent=2)
        params_summary = json.dumps(current_parameters, indent=2)
        
        # Construir prompt estructurado
        prompt = f"""
Analiza estos datos de trading y recomienda ajustes específicos a los parámetros de la estrategia de scalping.

## DATOS DE MERCADO ACTUALES
```json
{market_summary}
```

## RENDIMIENTO HISTÓRICO
```json
{performance_summary}
```

## PARÁMETROS ACTUALES
```json
{params_summary}
```

Responde ÚNICAMENTE en formato JSON con esta estructura exacta:
```json
{{
  "analysis": "Breve análisis de condiciones actuales (máx 100 palabras)",
  "parameter_adjustments": {{
    "param1": new_value,
    "param2": new_value
  }},
  "reasoning": "Explicación breve de por qué estos cambios (máx 150 palabras)",
  "market_trend": "alcista|bajista|lateral",
  "risk_adjustment": "aumentar|mantener|reducir",
  "confidence": 0.1-1.0
}}
```
No incluyas ningún texto adicional fuera del JSON.
"""
        return prompt
    
    def _build_signal_prompt(self, data_str: str, symbol: str, timeframe: str) -> str:
        """Construye prompt para generación de señales de trading"""
        prompt = f"""
Analiza los siguientes datos de mercado para {symbol} en timeframe {timeframe} y genera señales de trading precisas.

## DATOS DE MERCADO
{data_str}

Basándote únicamente en estos datos, genera un análisis técnico y una señal de trading.
Responde SOLO en formato JSON con la siguiente estructura exacta:
```json
{{
  "signal": "buy|sell|neutral",
  "entry_price": float,
  "stop_loss": float,
  "take_profit": float,
  "risk_reward_ratio": float,
  "confidence": float,  # entre 0.0 y 1.0
  "timeframe": "{timeframe}",
  "analysis": "Breve análisis que justifica la señal (máx 100 palabras)",
  "indicators": {{
    "indicador1": "señal o valor",
    "indicador2": "señal o valor"
  }}
}}
```
"""
        return prompt
    
    def _extract_suggestions(self, response_text: str) -> Dict[str, Any]:
        """Extrae sugerencias del texto de respuesta"""
        return self._extract_json(response_text)
    
    def _extract_signals(self, response_text: str) -> Dict[str, Any]:
        """Extrae señales de trading del texto de respuesta"""
        return self._extract_json(response_text)
    
    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        """Extrae JSON del texto de respuesta"""
        try:
            # Buscar el bloque JSON en la respuesta
            import re
            
            # Intentar encontrar JSON en la respuesta
            json_match = re.search(r'```json\s*(.*?)\s*```', 
                                   response_text, 
                                   re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Si no encuentra formato con ```json, intentar parsear todo
                json_str = response_text
            
            # Parsear JSON
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"Error extrayendo JSON: {str(e)}")
            return {"error": f"No se pudo extraer JSON válido: {str(e)}"}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula el indicador RSI para una serie de precios"""
        # Calcular cambios en los precios
        deltas = prices.diff()
        
        # Separar ganancias y pérdidas
        gain = deltas.where(deltas > 0, 0)
        loss = -deltas.where(deltas < 0, 0)
        
        # Calcular promedio de ganancias y pérdidas
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calcular RS y RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi