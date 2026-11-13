/**
 * PZF (PYME Z-compressed Format) decoder
 * Ported from pzf.js
 */

export function decodePZF(buffer) {
  const dv = new DataView(buffer)
  const fmtId = new TextDecoder("utf-8").decode(new Uint8Array(buffer, 0, 2))

  if (fmtId !== "BD") {
    throw new Error("Invalid PZF format")
  }

  const version = dv.getUint8(2)
  const fmt = dv.getUint8(3) // 0: u1, 1: u2, 2: f4
  const comp = dv.getUint8(4) // 0: raw, 1: huffcode, 2: chunked huffcode
  const quant = dv.getUint8(5) // 0: unquantized, 1: sqrt quantized
  const dimOrder = dv.getUint8(6) // 'C' or 'F'

  // RESERVED (1 byte)
  // SequenceID = int64 (offset=8)
  // FrameNum = int32 (offset=16)

  const width = dv.getUint32(20, true)
  const height = dv.getUint32(24, true)
  const depth = dv.getInt32(28, true)

  // FrameTimestamp = uint64, offset=32

  const quantOffset = dv.getFloat32(40, true)
  const quantScale = dv.getFloat32(44, true)

  let dataOffset
  if (version >= 3) {
    dataOffset = dv.getUint32(48, true)
  } else {
    dataOffset = 48
  }

  let data = new Uint8Array(buffer, dataOffset)

  if (comp !== 0) {
    console.warn('Data is compressed, decompression not yet implemented')
    // TODO: Implement huffman_decompress if needed
    throw new Error('Compressed PZF data not supported yet')
  }

  if (quant === 1) {
    // Data is quantized, dequantize
    data = new Float32Array(data.length)
    const u8Data = new Uint8Array(buffer, dataOffset)
    
    for (let i = 0; i < u8Data.length; i++) {
      const d = u8Data[i] * quantScale
      data[i] = d * d + quantOffset
    }

    switch (fmt) {
      case 0:
        // uint8
        data = new Uint8Array(data)
        break
      case 1:
        // uint16
        data = new Uint16Array(data)
        break
      case 2:
        // float32
        break
      default:
        console.error('Unrecognised data type')
    }
  } else {
    switch (fmt) {
      case 0:
        // uint8
        break
      case 1:
        // uint16
        data = new Uint16Array(data.buffer, data.byteOffset)
        break
      case 2:
        // float32
        data = new Float32Array(data.buffer, data.byteOffset)
        break
      default:
        console.error('Unrecognised data type')
    }
  }

  return {
    data: data,
    width: width,
    height: height,
    depth: depth,
    format: fmt,
    compressed: comp,
    quantized: quant
  }
}

/**
 * Map array data to RGBA with intensity scaling
 */
export function mapArrayToRGBA(data, cmin, cmax) {
  const out = new Uint8ClampedArray(data.length * 4)
  
  let min = Infinity
  let max = -Infinity

  for (let j = 0; j < data.length; j++) {
    const k = j * 4
    let v = data[j]
    
    min = Math.min(v, min)
    max = Math.max(v, max)
    
    v = (v - cmin) / (cmax - cmin)
    const v_ = 255 * v // simple grayscale map
    
    out[k] = v_       // R
    out[k + 1] = v_   // G
    out[k + 2] = v_   // B
    out[k + 3] = 255  // A
  }

  return {
    imageData: out,
    min: min,
    max: max
  }
}
