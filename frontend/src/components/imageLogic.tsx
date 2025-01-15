import { ImageLoadInfo } from "./interfaces"

const UTIF = require("./UTIF.js");

const checkPhases = (arr: Uint8ClampedArray, nChannels: number = 4) => {
    // given 2D image data, find N unique values, if > 6 call it unsegmented
    // ASSUMES RGBA ARRAYS: valid as UTIF decodes to RGBA
    const uniqueColours = arr.filter((_, i, __) => { return i % nChannels == 0 })
    const uniqueValues = [... new Set(uniqueColours)].sort(); // create Set (only unique vals) then unpack into arr
    const nPhases = uniqueValues.length;
    const segmented = (nPhases < 6) ? true : false;
    // TODO: if data RGBA then opacity will be counted as phase - this is bug
    // need to explictly look for unique rgb values
    return { nPhases: nPhases, segmented: segmented, vals: uniqueValues };
}

const findNChannels = (arr: Uint8ClampedArray, ih: number, iw: number) => {
    return Math.round(arr.length / (ih * iw))
}

export const replaceGreyscaleWithColours = (arr: Uint8ClampedArray, mapping: { [greyVal: number]: Array<number> }, nChannels: number = 4) => {
    const nPixels = Math.floor(arr.length / nChannels);
    const out = new Uint8ClampedArray(nPixels * 4).fill(0);
    for (let i = 0; i < nPixels; i = i + 1) {
        const queryVal = arr[nChannels * i];
        if (queryVal in mapping) {
            const [R, G, B, A] = mapping[queryVal];
            out[4 * i] = R;
            out[4 * i + 1] = G;
            out[4 * i + 2] = B;
            out[4 * i + 3] = A;
        }
    }
    return out;
}

export const getPhaseFraction = (arr: Uint8ClampedArray, val: number, nChannels: number = 4) => {
    console.log('ahhhh')
    const uniqueVals = arr.filter((_, i, __) => { return i % nChannels == 0 })
    const matching = uniqueVals.filter((v) => v == val);
    if (arr.length == 0) {return 0}
    return (matching.length) / (arr.length / nChannels);
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
        depth: tifs.length,
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
    const nChannels = findNChannels(imgData.data, img.height, img.width)
    console.log('n channels:' + String(nChannels))
    const phaseCheck = checkPhases(imgData.data, nChannels);

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
        depth: 1
    };
}