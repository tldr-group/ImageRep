import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { ImageLoadInfo, AnalysisInfo } from "./components/interfaces";

import Topbar from "./components/Topbar";
import DragDrop from "./components/DragDrop";
import PreviewCanvas from "./components/Canvas";
import Menu from "./components/Modals";

import { loadFromTIFF, loadFromImage } from "./components/imageLogic";

import "./assets/scss/App.scss";
import 'bootstrap/dist/css/bootstrap.min.css';

const PATH = "http://127.0.0.1:5000";
const PF_ENDPOINT = PATH + "/phasefraction";

const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 500; // 500MB

const App = () => {
    const {
        imageInfo: [imageInfo, setImageInfo],
        previewImg: [previewImg, setPreviewImg],
        selectedPhase: [selectedPhase,],
        targetL: [targetL, setTargetL],
        accurateFractions: [, setAccurateFractions],
        analysisInfo: [, setAnalysisInfo],
        menuState: [menuState, setMenuState],
    } = useContext(AppContext)!

    const appLoadFile = async (file: File) => {
        const reader = new FileReader();
        const extension = file.name.split('.').pop()?.toLowerCase();

        const isTIF = (extension === "tif" || extension === "tiff");
        const isPNGJPG = (extension === "png" || extension === "jpg" || extension === "jpeg");
        // need to parse file as arr buffer for tiff but href for png/jpeg etc. 
        if (isTIF) {
            reader.readAsArrayBuffer(file);
        } else if (isPNGJPG) {
            reader.readAsDataURL(file);
        } else {
            console.log(`Unsupported file format .${extension}`);
            return;
        };

        reader.onload = async () => {
            let result: ImageLoadInfo | null = null;
            if (file.size > MAX_FILE_SIZE_BYTES) {
                console.log(`File .${file.size / (1000 * 1000)}MB greater than max size (500MB)`);
                return;
            }

            if (isTIF) {
                result = loadFromTIFF(reader.result as ArrayBuffer);
            } else if (isPNGJPG) {
                const href = reader.result as string;
                result = await loadFromImage(href);
            };
            console.log(result);

            requestPhaseFraction(file);

            if (result?.segmented == false) {
                console.log('error: unsegmented');
            } else {
                result!.file = file;
                setImageInfo(result);
                setPreviewImg(result!.previewImg);
                setMenuState('phase');
            }
        };
    }

    const requestPhaseFraction = async (file: File) => {
        const formData = new FormData();
        formData.append('userFile', file);
        //formData.append('phaseVal', String(selectedPhaseValue));
        const resp = await fetch(PF_ENDPOINT, { method: 'POST', body: formData });
        const obj = await resp.json();
        const fractions = obj["phase_fractions"] as { [val: number]: number };
        console.log(fractions);
        setAccurateFractions(fractions);
    }

    useEffect(() => { // TODO: fetch from API instead
        if (menuState === 'processing') {
            setMenuState('conf_result');
            setTargetL(4 * imageInfo?.width!);
            setAnalysisInfo({
                integralRange: 1,
                z: 1,
                percentageErr: 10,
                absError: 5 * 0.45,
                lForDefaultErr: 4 * imageInfo?.width!,
                vf: 0.45
            })
        }
    }, [menuState])

    return (
        <div className={`w-full h-full`}>
            <Topbar></Topbar>
            <div className={`flex`} style={{ margin: '1.5%' }} > {/*Canvas div on left, sidebar on right*/}
                {!previewImg && <DragDrop loadFromFile={appLoadFile} />}
                {previewImg && <PreviewCanvas />}
            </div>
            <Menu></Menu>
        </div>
    );
};

export default App;