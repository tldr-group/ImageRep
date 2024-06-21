import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { imageLoadInfo } from "./components/interfaces";

import Topbar from "./components/Topbar"
import DragDrop from "./components/DragDrop";

import { loadFromTIFF, loadFromImage } from "./components/imageLogic";

import "./assets/scss/App.scss";
import 'bootstrap/dist/css/bootstrap.min.css';

const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 500; // 500MB


const loadFile = async (file: File) => {
    const reader = new FileReader();
    const extension = file.name.split('.').pop()?.toLowerCase();

    const isTIF = (extension === "tif" || extension === "tiff");
    const isPNGJPG = (extension === "png" || extension === "jpg" || extension === "jpeg");

    let result: imageLoadInfo | null = null;

    if (isTIF) {
        reader.readAsArrayBuffer(file); // array buffer for tif
    } else if (isPNGJPG) {
        reader.readAsDataURL(file); // href for png jpeg
    } else {
        console.log(`Unsupported file format .${extension}`);
        return;
    };

    reader.onload = async () => {
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
    };
}


const App = () => {
    const {
        image: [image, setImage],
    } = useContext(AppContext)!

    const foo = () => { }
    const appLoadFile = async (file: File) => {
        const res = await loadFile(file)
    }

    useEffect(() => { }, [])

    return (
        <div className={`w-full h-full`}>
            <Topbar></Topbar>
            <div className={`flex`} style={{ margin: '1.5%' }} > {/*Canvas div on left, sidebar on right*/}
                {!image && <DragDrop loadDefault={foo} loadFromFile={appLoadFile} />}
            </div>
        </div>
    );
};

export default App;