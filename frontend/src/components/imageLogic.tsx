import { ImageLoadInfo } from "./interfaces"

const UTIF = require("./UTIF.js");

const checkPhases = (arr: Uint8ClampedArray) => {
    // given 2D image data, find N unique values, if > 6 call it unsegmented
    // ASSUMES RGBA ARRAYS: valid as UTIF decodes to RGBA
    const uniqueColours = arr.filter((_, i, __) => { return i % 4 == 0 })
    const uniqueValues = [... new Set(uniqueColours)]; // create Set (only unique vals) then unpack into arr
    const nPhases = uniqueValues.length;
    const segmented = (nPhases < 6) ? true : false;
    // TODO: if data RGBA then opacity will be counted as phase - this is bug
    // need to explictly look for unique rgb values
    return { nPhases: nPhases, segmented: segmented, vals: uniqueValues };
}

export const replaceGreyscaleWithColours = (arr: Uint8ClampedArray, mapping: { [greyVal: number]: Array<number> }) => {
    const nPixels = Math.floor(arr.length / 4);
    const out = new Uint8ClampedArray(nPixels * 4).fill(0);
    for (let i = 0; i < arr.length; i = i + 4) {
        const queryVal = arr[i];
        if (queryVal in mapping) {
            const [R, G, B, A] = mapping[queryVal];
            out[i] = R;
            out[i + 1] = G;
            out[i + 2] = B;
            out[i + 3] = 255;
        }
    }
    return out;
}

export const getPhaseFraction = (arr: Uint8ClampedArray, val: number) => {
    const uniqueVals = arr.filter((_, i, __) => { return i % 4 == 0 })
    const matching = uniqueVals.filter((v) => v == val);
    return (100 * matching.length) / (arr.length / 4);
}

export const loadFromTIFF = (tiffBuffer: ArrayBuffer): ImageLoadInfo => {
    const tifs: Array<any> = UTIF.decode(tiffBuffer);
    const tif = tifs[0];
    // this needs to be done in-place before we can read the data
    UTIF.decodeImage(tiffBuffer, tif);

    const imgDataArr = new Uint8ClampedArray(UTIF.toRGBA8(tif));
    const imgData = new ImageData(imgDataArr, tif.width, tif.height);
    const img = getImagefromImageData(imgData, tif.height, tif.width);

    const nDims = (tifs.length > 1) ? 3 : 2;

    const phaseCheck = checkPhases(imgDataArr);

    return {
        file: null,
        previewData: imgData,
        previewImg: img,
        nDims: nDims,
        nPhases: phaseCheck.nPhases,
        phaseVals: phaseCheck.vals,
        segmented: phaseCheck.segmented,
        height: tif.height,
        width: tif.width,
    };
}

const getImageDataFromImage = (image: HTMLImageElement): ImageData => {
    // create temp canvas, draw image, get image data from canvas and return
    const tmpCanvas = document.createElement("canvas");
    const tmpContext = tmpCanvas.getContext("2d")!;

    tmpCanvas.width = image.width;
    tmpCanvas.height = image.height;
    tmpContext.drawImage(image, 0, 0);

    const data = tmpContext.getImageData(0, 0, image.width, image.height);
    tmpCanvas.remove();
    return data;
}

export const getImagefromImageData = (imageData: ImageData, height: number, width: number): HTMLImageElement => {
    const tmpCanvas = document.createElement("canvas");
    const tmpContext = tmpCanvas.getContext("2d")!;

    tmpCanvas.height = height;
    tmpCanvas.width = width;
    tmpContext.putImageData(imageData, 0, 0);

    const img = new Image(width, height);
    img.src = tmpCanvas.toDataURL();
    tmpCanvas.remove();
    return img;
}

export const loadFromImage = async (href: string): Promise<ImageLoadInfo> => {
    // load href to Image element, draw to temp canvas to extract pixel data
    const img = new Image();
    img.src = href;
    // decode ~ 'promisified onload handler' i.e blocks until source loaded
    await img.decode();

    const imgData = getImageDataFromImage(img);
    const phaseCheck = checkPhases(imgData.data);

    return {
        file: null,
        previewData: imgData,
        previewImg: img,
        nDims: 2,
        nPhases: phaseCheck.nPhases,
        phaseVals: phaseCheck.vals,
        segmented: phaseCheck.segmented,
        height: img.height,
        width: img.width,
    };
}