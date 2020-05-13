/**
 * Created by david on 12/04/20.
 */
function decode_pzf(buffer){
        dv = new DataView(buffer);
        fmt_id = new TextDecoder("utf-8").decode(new Uint8Array(buffer, 0, 2));

        if (fmt_id != "BD"){throw "Invalid PZF format"}
        //console.log(fmt_id);
        version = dv.getUint8(2);
        fmt = dv.getUint8(3); //0 : u1, 1:u2, 2 : f4

        comp = dv.getUint8(4); // 0: raw, 1 : huffcode, 2 : chuncked huffcode
        quant = dv.getUint8(5); // 0: unquantized, 1: sqrt quantized
        dim_order = dv.getUint8(6); //'C' or 'F'

        //RESERVED (1 byte)
        //SequenceID = int64 (offset=8)
        //FrameNum = int32 (offset =16)

        width = dv.getUint32(20, true);
        height = dv.getUint32(24, true);
        depth = dv.getInt32(28, true);

        //FrameTimestamp = uint64, offset= 32

        quant_offset = dv.getFloat32(40, true);
        quant_scale = dv.getFloat32(44, true);

        if (version >= 3){
            data_offset = dv.getUint32(48, true);
        } else {
            data_offset = 48;
        }

        data = new Uint8Array(buffer, data_offset);
        //console.log(data);

        if (comp != 0){
            console.log(comp);
            console.log('Data is compressed, decompression not yet implemented, so following will fail');
            data = huffman_decompress(data);
        }

        if (quant == 1){
            //Data is quantized, dequantize
            data = new Float32Array(data);
            data = data.map(function(d){
                d= d*quant_scale;
                return d*d + quant_offset;
            });

            switch(fmt){
                case 0:
                    //uint8
                    data = new Uint8Array(data);
                    break;
                case 1:
                    //uint16
                    data = new Uint16Array(data);
                    break;
                case 2:
                    //float32
                    break;
                default:
                    console.log('Unrecognised data type')
            }
        } else {
            switch(fmt) {
                case 0:
                    //uint8
                    break;
                case 1:
                    //uint16
                    //console.log(data.buffer, data.byteOffset);
                    data = new Uint16Array(data.buffer, data.byteOffset);
                    break;
                case 2:
                    //float32
                    data = new Float32Array(data.buffer, data.byteOffset);
                    break;
                default:
                    console.log('Unrecognised data type')
            }
        }

        //console.log(data);
        return {
            data: data,
            width: width,
            height: height,
            depth: depth
            // TODO - add more of the metadata
        }
    }