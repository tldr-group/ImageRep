import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { imageLoadInfo } from "./components/interfaces";

import Topbar from "./components/Topbar";
import DragDrop from "./components/DragDrop";
import PreviewCanvas from "./components/Canvas";

import { loadFromTIFF, loadFromImage } from "./components/imageLogic";

import "./assets/scss/App.scss";
import 'bootstrap/dist/css/bootstrap.min.css';

const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 500; // 500MB

const App = () => {
    const {
        previewData: [previewData, setPreviewData],
        userFile: [userFile, setUserFile],
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
            let result: imageLoadInfo | null = null;
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

            if (result?.segmented == false) {
                console.log('error: unsegmented');
            } else {
                setUserFile(file)
                setPreviewData(result!.previewData)

                // set data -> trigger user phase selection prompt
            }
        };
    }

    return (
        <div className={`w-full h-full`}>
            <Topbar></Topbar>
            <div className={`flex`} style={{ margin: '1.5%' }} > {/*Canvas div on left, sidebar on right*/}
                {!previewData && <DragDrop loadFromFile={appLoadFile} />}
                {previewData && <PreviewCanvas />}
            </div>
        </div>
    );
};

export default App;