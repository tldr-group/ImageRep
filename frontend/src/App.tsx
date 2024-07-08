import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { ImageLoadInfo, AnalysisInfo, IR_LIMIT_PX } from "./components/interfaces";

import Topbar from "./components/Topbar";
import DragDrop from "./components/DragDrop";
import PreviewCanvas from "./components/Canvas";
import { Menu } from "./components/Menu";
import { ErrorMessage, CLSModal, MoreInfo } from "./components/Popups"

import { loadFromTIFF, loadFromImage } from "./components/imageLogic";

import "./assets/scss/App.scss";
import 'bootstrap/dist/css/bootstrap.min.css';

const PATH = "http://127.0.0.1:5000";
const PF_ENDPOINT = PATH + "/phasefraction";
const REPR_ENDPOINT = PATH + "/repr";

const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 500; // 500MB

const App = () => {
    const {
        imageInfo: [imageInfo, setImageInfo],
        previewImg: [previewImg, setPreviewImg],
        selectedPhase: [selectedPhase, setSelectedPhase],
        selectedConf: [selectedConf, setSelectedConf],
        errVF: [errVF, setErrVF],
        targetL: [targetL, setTargetL],
        accurateFractions: [, setAccurateFractions],
        analysisInfo: [, setAnalysisInfo],
        menuState: [menuState, setMenuState],
        errorState: [errorState, setErrorState],
        showWarning: [showWarning, setShowWarning]
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
            setErrorState({ msg: `Unsupported file format .${extension}`, stackTrace: "" })
            return;
        };

        reader.onload = async () => {
            let result: ImageLoadInfo | null = null;
            if (file.size > MAX_FILE_SIZE_BYTES) {
                setErrorState({ msg: `File too large!`, stackTrace: `File .${file.size / (1000 * 1000)}MB greater than max size (500MB)` })
                return;
            }

            if (isTIF) {
                result = loadFromTIFF(reader.result as ArrayBuffer);
            } else if (isPNGJPG) {
                const href = reader.result as string;
                result = await loadFromImage(href);
            };
            console.log(result);

            if (result == null) {
                setErrorState({
                    msg: "Failed to load data",
                    stackTrace: ""
                });
                return;
            }

            if (result.segmented == false) {
                setErrorState({
                    msg: "Data is unsegmented - try using our web segmentation tool, SAMBA (https://www.sambasegment.com/)",
                    stackTrace: "Number of unique values > 6"
                });
                return;
            } else {
                if (result.height < 200 || result.width < 200 || (result.depth < 200 && result.depth > 1)) {
                    setShowWarning("size");
                }

                requestPhaseFraction(file);
                result.file = file;
                setImageInfo(result);
                setPreviewImg(result.previewImg);
                setMenuState('phase');
            }
        };
    }

    const requestPhaseFraction = async (file: File) => {
        try {
            const formData = new FormData();
            formData.append('userFile', file);
            //formData.append('phaseVal', String(selectedPhaseValue));
            const resp = await fetch(PF_ENDPOINT, { method: 'POST', body: formData });
            const obj = await resp.json();
            const fractions = obj["phase_fractions"] as { [val: number]: number };
            setAccurateFractions(fractions);
        } catch (e) {
            const error = e as Error;
            setErrorState({ msg: "Couldn't fetch phase fractions: data wrong or server down.", stackTrace: error.toString() });
        }
    }

    const requestRepr = async () => {
        try {
            const info = imageInfo!

            const formData = new FormData();
            formData.append('userFile', info.file!);
            formData.append('selected_phase', String(info.phaseVals[selectedPhase - 1]));
            formData.append('selected_conf', String(selectedConf));
            formData.append('selected_err', String(errVF));

            const resp = await fetch(REPR_ENDPOINT, { method: 'POST', body: formData });
            const obj = await resp.json();

            setMenuState('conf_result');
            setAnalysisInfo({
                integralRange: obj["cls"],
                z: 1,
                percentageErr: obj["percent_err"],
                absError: obj["abs_err"],
                lForDefaultErr: obj["l"],
                vf: 1
            })

            if (obj["cls"] > IR_LIMIT_PX) { setShowWarning("cls") }
            const minSide = Math.min(imageInfo?.width!, imageInfo?.height!)
            if (obj["l"] < minSide) { setShowWarning("over") }

            setTargetL(obj["l"]);
        } catch (e) {
            const error = e as Error;
            setErrorState({ msg: "Couldn't determine representativity: data wrong or server down.", stackTrace: error.toString() });
        }
    }

    const reset = () => {
        setMenuState('hidden');
        setPreviewImg(null);
        setImageInfo(null);
        setAnalysisInfo(null);
        setTargetL(null);
        setAccurateFractions(null);
        setPreviewImg(null);
        setSelectedPhase(0);
        setErrVF(5);
        setSelectedConf(95);
    }

    useEffect(() => { // TODO: fetch from API instead
        if (menuState === 'processing') {
            requestRepr();
        }
    }, [menuState])

    return (
        <div className={`w-full h-full`}>
            <Topbar loadFromFile={appLoadFile} reset={reset}></Topbar>
            <div className={`flex`} style={{ margin: '1.5%' }} > {/*Canvas div on left, sidebar on right*/}
                {!previewImg && <DragDrop loadFromFile={appLoadFile} />}
                {previewImg && <PreviewCanvas />}
            </div>
            <Menu />
            <ErrorMessage />
            {showWarning != "" && <CLSModal />}
            <MoreInfo />
        </div>
    );
};

export default App;