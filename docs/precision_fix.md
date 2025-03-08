# Solución: Error de Precisión en Órdenes de Binance

## Problema Identificado

Al ejecutar operaciones en Binance, el bot fallaba con el siguiente error:

```
APIError(code=-1111): Parameter 'quantity' has too much precision.
```

Este error ocurre porque la cantidad de criptomoneda a comprar (en este caso ETH) tenía demasiados decimales. Cada par de trading en Binance tiene reglas específicas sobre la precisión permitida para la cantidad (LOT_SIZE) y el precio (PRICE_FILTER).

## Solución Implementada

Se implementaron las siguientes mejoras en el archivo `binance_client.py`:

1. **Nueva función `get_symbol_info()`**: Obtiene las reglas de trading específicas para cada símbolo, incluyendo precisión de cantidad y precio.

2. **Nueva función `normalize_quantity()`**: Ajusta automáticamente la cantidad a la precisión adecuada según las reglas de Binance.

3. **Actualización de `create_spot_order()`**: Ahora normaliza automáticamente la cantidad antes de enviar la orden.

4. **Actualización de `create_order_with_sl_tp()`**: Normaliza la cantidad principal y en las órdenes de stop loss y take profit.

## Detalles Técnicos

El problema específico está relacionado con cómo Binance requiere que las cantidades cumplan con el parámetro `stepSize` del filtro `LOT_SIZE`. Por ejemplo:

- Si el `stepSize` es 0.00001, la cantidad debe tener como máximo 5 decimales
- Si el `stepSize` es 0.001, la cantidad debe tener como máximo 3 decimales

La solución:
1. Consulta los metadatos del símbolo (o usa valores conocidos en modo simulación)
2. Extrae el `stepSize` del filtro `LOT_SIZE`
3. Calcula la precisión decimal requerida
4. Trunca y formatea la cantidad según esa precisión

## Ejemplos

Para ETH:
- Cantidad original: 0.22168131980190559
- Cantidad normalizada: 0.22168

Para BTC:
- Cantidad original: 0.05234567891
- Cantidad normalizada: 0.052345

## Beneficios Adicionales

1. **Depuración mejorada**: Se agregaron mensajes de log detallados en modo debug que muestran tanto la cantidad original como la normalizada.

2. **Robustez mejorada**: Esta solución previene errores similares para todos los símbolos de trading, no solo para ETHUSDT.

3. **Simulación realista**: El modo de simulación ahora imita correctamente las restricciones de precisión de Binance.

## Cómo Funciona

Cuando se llama a `create_order_with_sl_tp()` o `create_spot_order()`, la cantidad se normaliza automáticamente:

```python
normalized_quantity = self.normalize_quantity(symbol, quantity)
```

La función `normalize_quantity()` hace:

1. Obtiene la información del símbolo con `get_symbol_info()`
2. Extrae el filtro `LOT_SIZE` y su `stepSize`
3. Calcula los decimales necesarios basados en `stepSize`
4. Trunca la cantidad para que sea un múltiplo de `stepSize`
5. Formatea la cantidad como string con la precisión exacta requerida

## Pruebas

Para probar que la solución funciona:

1. Ejecuta el bot con el flag `--debug` para ver los logs detallados:
   ```
   python main.py --symbols BTCUSDT,ETHUSDT --debug
   ```

2. Verifica en los logs que las cantidades se están normalizando correctamente:
   ```
   DEBUG: Original quantity for ETHUSDT: 0.22168131980190559, normalized: 0.22168
   ```