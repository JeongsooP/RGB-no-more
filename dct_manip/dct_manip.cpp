#include <stdio.h>
#include <jpeglib.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <iostream>
#include <memory>
#include <cmath>

#include <pybind11/embed.h>
#include <torch/extension.h>

/*
Many of these functions are from TorchJPEG, codec_ops.cpp
All thanks to the original authors:
Max Ehrlich, Larry Davis, Ser-Nam Lim, and Abhinav Shrivastava. “Quantization Guided JPEG Artifact Correction.” In Proceedings of the European Conference on Computer Vision, 2020
https://queuecumber.gitlab.io/torchjpeg/
*/

class libjpeg_exception : public std::exception {
private:
    char *error;

public:
    libjpeg_exception(j_common_ptr cinfo) {
        error = new char[JMSG_LENGTH_MAX];
        (cinfo->err->format_message)(cinfo, error);
    }

    virtual const char *what() const throw() {
        return error;
    }
};

void raise_libjpeg(j_common_ptr cinfo) {
    throw libjpeg_exception(cinfo);
}

long jdiv_round_up(long a, long b)
/* Compute a/b rounded up to next integer, ie, ceil(a/b) */
/* Assumes a >= 0, b > 0 */
{
    return (a + b - 1L) / b;
}

extern "C" {
// On some machines, this free will be name-mangled if it isn't in extern "C" here
void free_buffer(unsigned char *buffer) {
    free(buffer);
}
}

torch::Tensor interleave_rgb(torch::Tensor data){ // C x H x W
    // interleaves rgb channels to a jpeg-compatible matrix. They have to be R,G,B, R,G,B, R,G,B, in memory (has to be in the same row)
    int C = data.size(0);
    int H = data.size(1);
    int W = data.size(2);

    auto interleaved = torch::empty({H, C*W}, data.dtype());
    auto data_acc = data.accessor<uint8_t, 3>(); // data accessor for fast accessing
    auto inter_acc = interleaved.accessor<uint8_t, 2>();

    for(int i=0; i<C; i++){
        for(int j=0; j<H; j++){
            for(int k=0; k<W; k++){
                inter_acc[j][i+C*k] = data_acc[i][j][k]; // interleaves RGB
            }
        }
    }

    return interleaved;
}

void extract_channel(const jpeg_decompress_struct &srcinfo, //srcinfo = cinfo
                     jvirt_barray_ptr *src_coef_arrays,
                     int compNum, // component number. 0=Y, 1=Cb, 2=Cr
                     torch::Tensor coefficients, // Tensor to write DCT coefficients
                     torch::Tensor quantization, // Tensor to write Quantization
                     int &coefficients_written) {
    for (JDIMENSION rowNum = 0; rowNum < srcinfo.comp_info[compNum].height_in_blocks; rowNum++) {
        JBLOCKARRAY rowPtrs = srcinfo.mem->access_virt_barray((j_common_ptr)&srcinfo, src_coef_arrays[compNum],
                                                              rowNum, 1, FALSE);

        for (JDIMENSION blockNum = 0; blockNum < srcinfo.comp_info[compNum].width_in_blocks; blockNum++) {
            std::copy_n(rowPtrs[0][blockNum], DCTSIZE2, coefficients.data_ptr<int16_t>() + coefficients_written); //copy_n : copies source[i:i+N) to dest[j:j+N) copy_n(source, N, dest)
            coefficients_written += DCTSIZE2; // num_written coefficients
        }
    }

    std::copy_n(srcinfo.comp_info[compNum].quant_table->quantval, DCTSIZE2,
                quantization.data_ptr<int16_t>() + DCTSIZE2 * compNum); // read quantization table and copy it to quantization pointer
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> read_coefficients_using(jpeg_decompress_struct &srcinfo) {
    jpeg_read_header(&srcinfo, TRUE); // read jpeg header

    // channels x 2
    auto dimensions = torch::empty({srcinfo.num_components, 2}, torch::kInt); // dimension for original file
    auto dct_dim_a = dimensions.accessor<int, 2>(); // accessor for accessing element of a tensor much more efficiently
    for (auto i = 0; i < srcinfo.num_components; i++) {
        dct_dim_a[i][0] = srcinfo.comp_info[i].downsampled_height;
        dct_dim_a[i][1] = srcinfo.comp_info[i].downsampled_width;
    }

    // read coefficients
    jvirt_barray_ptr *src_coef_arrays = jpeg_read_coefficients(&srcinfo);

    auto Y_coefficients = torch::empty({1,
                                        srcinfo.comp_info[0].height_in_blocks,
                                        srcinfo.comp_info[0].width_in_blocks,
                                        DCTSIZE,
                                        DCTSIZE},
                                       torch::kShort); // (1, (H/8), (W/8), 8, 8)

    auto quantization = torch::empty({srcinfo.num_components, DCTSIZE, DCTSIZE}, torch::kShort); // (3, 8, 8)

    // extract Y channel
    int cw = 0;
    // srcinfo = cinfo of jpeg_decompress struct, src_coef_array: jpeg_read_coefficients, 0: current component num, Y_coefficients, quantization: destination array, cw: coefficients_written (tracking where to write next)
    extract_channel(srcinfo, src_coef_arrays, 0, Y_coefficients, quantization, cw); 

    // extract CrCb channels
    auto CrCb_coefficients = std::optional<torch::Tensor>{};

    if (srcinfo.num_components > 1) {
        CrCb_coefficients = torch::empty({2,
                                          srcinfo.comp_info[1].height_in_blocks,
                                          srcinfo.comp_info[1].width_in_blocks,
                                          DCTSIZE,
                                          DCTSIZE},
                                         torch::kShort); // (2, (H/16), (W/16), 8, 8)

        cw = 0;
        extract_channel(srcinfo, src_coef_arrays, 1, *CrCb_coefficients, quantization, cw);
        extract_channel(srcinfo, src_coef_arrays, 2, *CrCb_coefficients, quantization, cw);
    }

    // cleanup
    jpeg_finish_decompress(&srcinfo);

    return {
        dimensions,
        quantization,
        Y_coefficients,
        CrCb_coefficients};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> read_coefficients(const std::string &path) {
    // open the file
    FILE *infile; //filestream (defined in jpeg)
    if ((infile = fopen(path.c_str(), "rb")) == nullptr) {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    // start decompression
    jpeg_decompress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_decompress(&cinfo); // initialization till this point

    jpeg_stdio_src(&cinfo, infile); // pass filestream

    auto ret = read_coefficients_using(cinfo); // custom function (defined above)

    jpeg_destroy_decompress(&cinfo); //close
    fclose(infile);

    return ret;
}

void set_quantization(j_compress_ptr cinfo, torch::Tensor quantization) {
    int num_components = quantization.size(0);
    std::copy_n(quantization.data_ptr<int16_t>(), DCTSIZE2, cinfo->quant_tbl_ptrs[0]->quantval);

    if (num_components > 1) {
        std::copy_n(quantization.data_ptr<int16_t>() + DCTSIZE2, DCTSIZE2, cinfo->quant_tbl_ptrs[1]->quantval);
    }
}

jvirt_barray_ptr *request_block_storage(j_compress_ptr cinfo) {
    auto block_arrays = (jvirt_barray_ptr *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo,
                                                                       JPOOL_IMAGE,
                                                                       sizeof(jvirt_barray_ptr *) *
                                                                           cinfo->num_components);

    std::transform(cinfo->comp_info, cinfo->comp_info + cinfo->num_components, block_arrays,
                   [&](jpeg_component_info &compptr) {
                       int MCU_width = jdiv_round_up((long)cinfo->jpeg_width, (long)compptr.MCU_width);
                       int MCU_height = jdiv_round_up((long)cinfo->jpeg_height, (long)compptr.MCU_height);

                       return (cinfo->mem->request_virt_barray)((j_common_ptr)cinfo,
                                                                JPOOL_IMAGE,
                                                                TRUE,
                                                                MCU_width,
                                                                MCU_height,
                                                                compptr.v_samp_factor);
                   });

    return block_arrays;
}

void fill_extended_defaults(j_compress_ptr cinfo, int color_samp_factor = 2) {

    cinfo->jpeg_width = cinfo->image_width;
    cinfo->jpeg_height = cinfo->image_height;

    jpeg_set_defaults(cinfo);

    cinfo->comp_info[0].component_id = 1;
    cinfo->comp_info[0].h_samp_factor = 1;
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[0].quant_tbl_no = 0;
    cinfo->comp_info[0].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE);
    cinfo->comp_info[0].height_in_blocks = jdiv_round_up(cinfo->jpeg_height, DCTSIZE);
    cinfo->comp_info[0].MCU_width = 1;
    cinfo->comp_info[0].MCU_height = 1;

    if (cinfo->num_components > 1) {
        cinfo->comp_info[0].h_samp_factor = color_samp_factor;
        cinfo->comp_info[0].v_samp_factor = color_samp_factor;
        cinfo->comp_info[0].MCU_width = color_samp_factor;
        cinfo->comp_info[0].MCU_height = color_samp_factor;

        for (int c = 1; c < cinfo->num_components; c++) {
            cinfo->comp_info[c].component_id = 1 + c;
            cinfo->comp_info[c].h_samp_factor = 1;
            cinfo->comp_info[c].v_samp_factor = 1;
            cinfo->comp_info[c].quant_tbl_no = 1;
            cinfo->comp_info[c].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor);
            cinfo->comp_info[c].height_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor);
            cinfo->comp_info[c].MCU_width = 1;
            cinfo->comp_info[c].MCU_height = 1;
        }
    }

    cinfo->min_DCT_h_scaled_size = DCTSIZE;
    cinfo->min_DCT_v_scaled_size = DCTSIZE;
}

void set_channel(const jpeg_compress_struct &cinfo,
                 torch::Tensor coefficients,
                 jvirt_barray_ptr *dest_coef_arrays,
                 int compNum,
                 int &coefficients_written) {
    for (JDIMENSION rowNum = 0; rowNum < cinfo.comp_info[compNum].height_in_blocks; rowNum++) {
        JBLOCKARRAY rowPtrs = cinfo.mem->access_virt_barray((j_common_ptr)&cinfo, dest_coef_arrays[compNum],
                                                            rowNum, 1, TRUE);

        for (JDIMENSION blockNum = 0; blockNum < cinfo.comp_info[compNum].width_in_blocks; blockNum++) {
            std::copy_n(coefficients.data_ptr<int16_t>() + coefficients_written, DCTSIZE2, rowPtrs[0][blockNum]);
            coefficients_written += DCTSIZE2;
        }
    }
}

void write_coefficients(const std::string &path,
                        torch::Tensor dimensions,
                        torch::Tensor quantization,
                        torch::Tensor Y_coefficients,
                        std::optional<torch::Tensor> CrCb_coefficients = std::nullopt) {
    FILE *outfile;
    if ((outfile = fopen(path.c_str(), "wb")) == nullptr) {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    jpeg_compress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo); // initialization to this point
    jpeg_stdio_dest(&cinfo, outfile); // pass filestream

    auto dct_dim_a = dimensions.accessor<int, 2>(); // accessing dimensions.

    cinfo.image_height = dct_dim_a[0][0]; // write dimensions to compress struct
    cinfo.image_width = dct_dim_a[0][1];
    cinfo.input_components = CrCb_coefficients ? 3 : 1;
    cinfo.in_color_space = CrCb_coefficients ? JCS_RGB : JCS_GRAYSCALE;

    fill_extended_defaults(&cinfo);

    set_quantization(&cinfo, quantization); // write quantization info

    jvirt_barray_ptr *coef_dest = request_block_storage(&cinfo);
    jpeg_write_coefficients(&cinfo, coef_dest); // prepare to write DCT blocks to cinfo

    int cw = 0;
    set_channel(cinfo, Y_coefficients, coef_dest, 0, cw); //write Y

    if (CrCb_coefficients) {
        cw = 0;
        set_channel(cinfo, *CrCb_coefficients, coef_dest, 1, cw); //write color if specified
        set_channel(cinfo, *CrCb_coefficients, coef_dest, 2, cw);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    fclose(outfile);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> quantize_at_quality(torch::Tensor pixels, int quality, bool baseline = true) {
    // Use libjpeg to compress the pixels into a memory buffer, this is slightly wasteful
    // as it performs entropy coding
    struct jpeg_compress_struct cinfo {
    };

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo); // init

    unsigned long compressed_size;
    unsigned char *buffer = nullptr;
    jpeg_mem_dest(&cinfo, &buffer, &compressed_size); // memory buffer destination

    cinfo.image_width = pixels.size(2); // write Width to cinfo
    cinfo.image_height = pixels.size(1); // write Height to cinfo
    cinfo.input_components = pixels.size(0); // write # of components (Y Cb Cr = 3) to cinfo
    cinfo.in_color_space = pixels.size(0) > 1 ? JCS_RGB : JCS_GRAYSCALE; // set colorspace

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, boolean(baseline));

    // No way that I know of to pass planar images to libjpeg //----- torchjpeg code generates inverted DCT coefficients actually.
    //auto channel_interleaved = (pixels * 255.f).round().to(torch::kByte).transpose(0, 2).transpose(0, 1).contiguous();
    auto channel_interleaved = interleave_rgb(pixels);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1]; 
    //while (cinfo.next_scanline < cinfo.image_height) { //----- torchjpeg code
    //    row_pointer[0] = channel_interleaved.data_ptr<JSAMPLE>() +
    //                     cinfo.next_scanline * channel_interleaved.size(1) * channel_interleaved.size(2);
    //    jpeg_write_scanlines(&cinfo, row_pointer, 1);
    //}
    int row_stride = channel_interleaved.size(1);
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = channel_interleaved.data_ptr<JSAMPLE>() + cinfo.next_scanline * row_stride; // get pointer and the corresponding row
        jpeg_write_scanlines(&cinfo, row_pointer, 1); // write row
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // Decompress memory buffer to DCT coefficients
    jpeg_decompress_struct srcinfo{};
    struct jpeg_error_mgr srcerr {
    };

    srcinfo.err = jpeg_std_error(&srcerr);
    jpeg_create_decompress(&srcinfo);

    jpeg_mem_src(&srcinfo, buffer, compressed_size);

    auto ret = read_coefficients_using(srcinfo);

    jpeg_destroy_decompress(&srcinfo);
    free_buffer(buffer);

    return ret;
}

void write_tensor(const std::string &path, // path to save file
                        torch::Tensor dataarr, // C x H x W
                        torch::Tensor quantization, // Quantization Table
                        int quality = 100) // compression quality
                        {
    // Write from bytestream & custom quantization table
    FILE *outfile;
    if ((outfile = fopen(path.c_str(), "wb")) == nullptr) {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    jpeg_compress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo); // initialization to this point
    jpeg_stdio_dest(&cinfo, outfile); // pass filestream

    cinfo.image_height = dataarr.size(1); // write dimensions to compress struct
    cinfo.image_width = dataarr.size(2);
    cinfo.input_components = dataarr.size(0);
    cinfo.in_color_space = (dataarr.size(0)==3) ? JCS_RGB : JCS_GRAYSCALE;

    fill_extended_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE); // set compression quality.
    set_quantization(&cinfo, quantization); // write quantization info

    jpeg_start_compress(&cinfo, TRUE); //start compression

    // Interleave RGB Pixels. They have to be R,G,B, R,G,B, R,G,B, in memory (has to be in the same row)
    auto channel_interleaved = interleave_rgb(dataarr);

    JSAMPROW row_pointer[1]; // pointer to JSAMPLE rows
    int row_stride = channel_interleaved.size(1);
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = channel_interleaved.data_ptr<JSAMPLE>() + cinfo.next_scanline * row_stride; // get pointer and the corresponding row
        jpeg_write_scanlines(&cinfo, row_pointer, 1); // write row
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    fclose(outfile);
}

torch::Tensor read_jpeg(const std::string &path){
    // Reads JPEG and returns Tensor
    // start decompression
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr; // jpeg error handler
    
    // open the file
    FILE *infile; //filestream (defined in jpeg)
    JSAMPARRAY buffer; //output row buffer
    int row_stride; // physical row width in output buffer
    if ((infile = fopen(path.c_str(), "rb")) == nullptr) {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg; // raise_libjpeg error handler

    jpeg_create_decompress(&cinfo); // initialize JPEG Decompression object

    jpeg_stdio_src(&cinfo, infile); // pass filestream // data source

    jpeg_read_header(&cinfo, TRUE); // read jpeg header

    // may set parameters here. Not needed (using default set by jpeg_read_header())
    // parameters may include: grayscale reading from color images, etc.

    jpeg_start_decompress(&cinfo); // start decompressor
    // initialize output tensor matrix
    auto decompressed = torch::empty({cinfo.output_components, // can only access these info after starting decompressor
                                      cinfo.output_height,
                                      cinfo.output_width},
                                      torch::kUInt8); // (1, (H/8), (W/8), 8, 8)
    auto decomp_acc = decompressed.accessor<uint8_t, 3>(); // accessor for quickly accessing decompressed array

    row_stride = cinfo.output_width * cinfo.output_components; // JSAMPLEs per row in output buffer
    buffer = (cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    JDIMENSION i; // iterator // peculiar unsigned int for jpeglib
    int j; // iterator

    // loop over scanlines
    while (cinfo.output_scanline < cinfo.output_height) { // cinfo.output_scanline tracks how many times you called jpeg_read_scanlines.
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (i=0;i<cinfo.output_width;i++){ // write buffer to tensor
            for (j=0;j<cinfo.output_components;j++){
                decomp_acc[j][cinfo.output_scanline-1][i] = *(buffer[0] + j + i*cinfo.output_components); // write to (C x H x W) tensor
            }
        }
    }

    jpeg_finish_decompress(&cinfo); // finish decompressing
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return decompressed; // return tensor
}

torch::Tensor decode_coeff(torch::Tensor dimensions,
                        torch::Tensor quantization,
                        torch::Tensor Y_coefficients,
                        std::optional<torch::Tensor> CrCb_coefficients = std::nullopt,
                        int quality = -1
                        ) {
    // Decode DCT coefficients to RGB Tensor.
    // Use "quality" when you cannot recover quantization table
    // you must be absolutely sure that your 'quality' generates the correct 'quantization' otherwise the recovered data would not be the same!
    struct jpeg_compress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo); // initialization to this point
    unsigned long compressed_size;
    unsigned char *buffer = nullptr;
    jpeg_mem_dest(&cinfo, &buffer, &compressed_size); // pass memory buffer

    auto dct_dim_a = dimensions.accessor<int, 2>(); // accessing dimensions.

    cinfo.image_height = dct_dim_a[0][0]; // write dimensions to compress struct
    cinfo.image_width = dct_dim_a[0][1];
    cinfo.input_components = CrCb_coefficients ? 3 : 1;
    cinfo.in_color_space = CrCb_coefficients ? JCS_RGB : JCS_GRAYSCALE;

    fill_extended_defaults(&cinfo);

    if(quality > 0){ // if quality is set
        jpeg_set_quality(&cinfo, quality, TRUE);
    }
    else{
        set_quantization(&cinfo, quantization); // write quantization info
    }

    jvirt_barray_ptr *coef_dest = request_block_storage(&cinfo);
    jpeg_write_coefficients(&cinfo, coef_dest); // prepare to write DCT blocks to cinfo

    int cw = 0;
    set_channel(cinfo, Y_coefficients, coef_dest, 0, cw); // write Y

    if (CrCb_coefficients) {
        cw = 0;
        set_channel(cinfo, *CrCb_coefficients, coef_dest, 1, cw); //write color if specified
        set_channel(cinfo, *CrCb_coefficients, coef_dest, 2, cw);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // Decompress memory buffer to DCT coefficients
    jpeg_decompress_struct srcinfo{};
    struct jpeg_error_mgr srcerr {
    };

    srcinfo.err = jpeg_std_error(&srcerr);
    jpeg_create_decompress(&srcinfo);

    jpeg_mem_src(&srcinfo, buffer, compressed_size);
    jpeg_read_header(&srcinfo, TRUE);

    jpeg_start_decompress(&srcinfo);

    JSAMPARRAY ROWBUFFER;
    int row_stride; // physical row width in output buffer
    JDIMENSION i; // iterator // peculiar unsigned int for jpeglib
    int j; // iterator

    auto decompressed = torch::empty({srcinfo.output_components, // can only access these info after starting decompressor
                                      srcinfo.output_height,
                                      srcinfo.output_width},
                                      torch::kUInt8); // (1, (H/8), (W/8), 8, 8)
    auto decomp_acc = decompressed.accessor<uint8_t, 3>(); // accessor for quickly accessing decompressed array

    row_stride = srcinfo.output_width * srcinfo.output_components; // JSAMPLEs per row in output buffer
    ROWBUFFER = (srcinfo.mem->alloc_sarray)((j_common_ptr)&srcinfo, JPOOL_IMAGE, row_stride, 1);
    while (srcinfo.output_scanline < srcinfo.output_height) { // cinfo.output_scanline tracks how many times you called jpeg_read_scanlines.
        jpeg_read_scanlines(&srcinfo, ROWBUFFER, 1);
        for (i=0;i<srcinfo.output_width;i++){ // write buffer to tensor
            for (j=0;j<srcinfo.output_components;j++){
                decomp_acc[j][srcinfo.output_scanline-1][i] = *(ROWBUFFER[0] + j + i*srcinfo.output_components); // write to (C x H x W) tensor
            }
        }
    }

    jpeg_finish_decompress(&srcinfo); // finish decompressing
    jpeg_destroy_decompress(&srcinfo);
    free_buffer(buffer);

    return decompressed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_coefficients", &read_coefficients, R"(
            read_coefficients(path: str) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]

            Read DCT coefficients from a JPEG file

            Parameters
            ----------
            path : str
                The path to an existing JPEG file

            Returns
            -------
            Tensor
                A :math:`\left(C, 2 \right)` Tensor containing the size of the original image that produced the returned DCT coefficients, this is usually different from the size of the
                coefficient Tensor because padding is added during the compression process. The format is :math:`\left(H, W \right)`.
            Tensor
                A :math:`\left(C, 8, 8 \right)` Tensor containing the quantization matrices for each of the channels. Usually the color channels have the same quantization matrix.
            Tensor
                A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Y channel DCT coefficients for each :math:`8 \times 8` block.
            Optional[Tensor]
                A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Cb and Cr channel DCT coefficients for each :math:`8 \times 8` block, or `None` if the image is grayscale.

            Note
            -----
            The return values from this function are "raw" values, as output by libjpeg with no transformation. In particular, the DCT coefficients are quantized and will need
            to be dequantized using the returned quantization matrices before they can be converted into displayable image pixels. They will likely also need cropping and the chroma
            channels, if they exist, will probably be downsampled. The type of all Tensors is :code:`torch.short` except the dimensions (first return value) with are of type :code:`torch.int`.
          )");
    m.def("write_coefficients", &write_coefficients, R"(
            write_coefficients(path: str, dimensions: Tensor, quantization: Tensor, Y_coefficients: Tensor, CrCb_coefficients: Optional[Tensor] = None) -> None

            Write DCT coefficients to a JPEG file.

            Parameters
            ----------
            path : str
                The path to the JPEG file to write, will be overwritten
            dimensions : Tensor
                A :math:`\left(C, 2 \right)` Tensor containing the size of the original image before taking the DCT. If you padded the image to produce the coefficients, pass the size before padding here.
            quantization : Tensor
                A :math:`\left(C, 8, 8 \right)` Tensor containing the quantization matrices that were used to quantize the DCT coefficients.
            Y_coefficients : Tensor
                A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of Y channel DCT coefficients separated into :math:`8 \times 8` blocks.
            CbCr_coefficients : Optional[Tensor]
                A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of Cb and Cr channel DCT coefficients separated into :math:`8 \times 8` blocks.

            Note
            -----
            The parameters passed to this function are in the same "raw" format as returned by :py:func:`read_coefficients`. The DCT coefficients must be appropriately quantized and the color 
            channel coefficients must be downsampled if desired. The type of the Tensors must be :code:`torch.short` except the :code:`dimensions` parameter which must be :code:`torch.int`.
          )",
          py::arg("path"), py::arg("dimensions"), py::arg("quantization"), py::arg("Y_coefficients"),
          py::arg("CrCb_coefficients") = std::nullopt);
    m.def("quantize_at_quality", &quantize_at_quality, R"(
            quantize_at_quality(pixels: Tensor, quality: int, baseline: bool = true) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]

            Quantize pixels using libjpeg at the given quality. By using this function instead of :py:mod:`torchjpeg.quantization` the result
            is guaranteed to be exactly the same as if the JPEG was quantized using an image library like Pillow and the coefficients are returned
            directly without needing to be recomputed from pixels.

            Parameters
            ----------
            pixels : Tensor
                A :math:`\left(C, H, W \right)` Tensor of image pixels in pytorch format (normalized to [0, 1]).
            quality : int
                The integer quality level to quantize to, in [0, 100] with 100 being maximum quality and 0 being minimal quality.
            baseline : bool
                Use the baseline quantization matrices, e.g. quantization matrix entries cannot be larger than 255. True by default, don't change it unless you know what you're doing.

            Returns
            -------
            Tensor
                A :math:`\left(C, 2 \right)` Tensor containing the size of the original image that produced the returned DCT coefficients, this is usually different from the size of the
                coefficient Tensor because padding is added during the compression process. The format is :math:`\left(H, W \right)`.
            Tensor
                A :math:`\left(C, 8, 8 \right)` Tensor containing the quantization matrices for each of the channels. Usually the color channels have the same quantization matrix.
            Tensor
                A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Y channel DCT coefficients for each :math:`8 \times 8` block.
            Optional[Tensor]
                A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Cb and Cr channel DCT coefficients for each :math:`8 \times 8` block, or `None` if the image is grayscale.

            Note
            -----
            The output format of this function is the same as that of :py:func:`read_coefficients`. 
          )",
          py::arg("pixels"), py::arg("quality"), py::arg("baseline") = true);
    m.def("write_tensor", &write_tensor, "Write JPEG file from Tensor (C x H x W). dtype must be uint8 (0~255)");
    m.def("read_jpeg", &read_jpeg, "Read JPEG file to Tensor (RGB or gray) (C x H x W) returns uint8 tensor");
    m.def("decode_coeff",&decode_coeff, "Decode DCT coefficients to RGB tensor",
    py::arg("dim"), py::arg("quant"), py::arg("y"), py::arg("cbcr"), py::arg("quality") = -1);
}