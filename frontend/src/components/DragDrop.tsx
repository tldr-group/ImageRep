import React, { useContext, useEffect, useRef, useState } from "react";
import { DragDropProps } from "./interfaces";

const DEFAULT_IMAGE_2D = "../assets/default_2D.tiff";
const DEFAULT_IMAGE_3D = "../assets/default_3D.tiff";

export const dragDropStyle = {
    height: '75vh', width: '75vw',
    outline: '10px dashed #b5bab6', color: '#b5bab6',
    fontSize: '3em', justifyContent: 'center', alignItems: 'center',
    borderRadius: '10px', padding: '10px', display: 'flex', margin: 'auto'
}
const flexCenterClasses = "flex items-center justify-center";

const isMobile = (window.innerWidth < 800)

const DragDrop = ({ loadFromFile }: DragDropProps): JSX.Element => {
    // Drag and drop for file upload
    const handleDrag = (e: any) => { e.preventDefault(); }
    const handeDrop = (e: any) => {
        e.preventDefault();
        if (e.dataTransfer.items) {
            const item = e.dataTransfer.items[0];
            if (item.kind === "file") {
                const file = item.getAsFile();
                loadFromFile(file);
            };
        };
    };

    const viewExample = async (path: string) => {
        const url = new URL(path, location.origin);
        console.log(url)
        const resp = await fetch(url);
        const data = await resp.blob();
        const metadata = { type: data.type }
        const file = new File([data], 'default.tiff', metadata)
        console.log(file)
        loadFromFile(file)
    }

    return (
        <div style={dragDropStyle}
            onDragOver={handleDrag}
            onDrop={handeDrop}
        >
            {(!isMobile) && <div style={{display: 'flex', alignItems: 'center', flexDirection: 'column'}}>
                <span>Drag microstructure file, or view example <a style={{ cursor: "pointer", color: 'blue' }} onClick={e => viewExample(DEFAULT_IMAGE_2D)}>in 2D</a> or  <a style={{ cursor: "pointer", color: 'blue' }} onClick={e => viewExample(DEFAULT_IMAGE_3D)}> 3D</a></span>
                <span style={{fontSize: '0.7em'}}>(image must be segmented & image background (e.g. scale bar) cropped out)</span>
            </div>}
        </div>
    );
}

export default DragDrop
