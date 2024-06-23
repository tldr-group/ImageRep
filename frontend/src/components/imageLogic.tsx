import { imageLoadInfo } from "./interfaces"

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
    return { nPhases: nPhases, segmented: segmented };
}

const remapImageDataArr = (arr: Uint8ClampedArray, originalValues: Array<number>, newValues: Array<number>) => {
    // given arr, list of original values in arr and desired new values, loop over arr and replace
    const nPixels = arr.length;
    const out = new Uint8ClampedArray(nPixels).fill(0);
    for (let i = 0; i < nPixels; i++) {
        const currentVal = arr[i];
        const idx = originalValues.indexOf(currentVal);
        out[i] = newValues[idx];
    }
    return out;
}

export const loadFromTIFF = (tiffBuffer: ArrayBuffer): imageLoadInfo => {
    const tifs: Array<any> = UTIF.decode(tiffBuffer);
    const tif = tifs[0];
    // this needs to be done in-place before we can read the data
    UTIF.decodeImage(tiffBuffer, tif);

    const imgDataArr = new Uint8ClampedArray(UTIF.toRGBA8(tif));
    const imgData = new ImageData(imgDataArr, tif.width, tif.height);
    const img = getImagefromImageData(imgData, tif.height, tif.width);

    const nDims = (tifs.length > 1) ? 3 : 2;

    const phaseCheck = checkPhases(imgDataArr);
    const nPhases = phaseCheck.nPhases;
    const segmented = phaseCheck.segmented;

    return {
        previewData: imgData,
        previewImg: img,
        nDims: nDims,
        nPhases: nPhases,
        segmented: segmented,
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

const getImagefromImageData = (imageData: ImageData, height: number, width: number): HTMLImageElement => {
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

export const loadFromImage = async (href: string): Promise<imageLoadInfo> => {
    // load href to Image element, draw to temp canvas to extract pixel data
    const img = new Image();
    img.src = href;
    // decode ~ 'promisified onload handler' i.e blocks until source loaded
    await img.decode();

    const imgData = getImageDataFromImage(img);
    const phaseCheck = checkPhases(imgData.data);
    const nPhases = phaseCheck.nPhases;
    const segmented = phaseCheck.segmented;

    return {
        previewData: imgData,
        previewImg: img,
        nDims: 2,
        nPhases: nPhases,
        segmented: segmented,
        height: img.height,
        width: img.width,
    };
}