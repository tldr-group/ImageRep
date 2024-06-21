import React, { useContext, useEffect, useRef, useState } from "react";


const centredStyle = {
    height: '60vh', width: '60vw',
    justifyContent: 'center', alignItems: 'center',
    padding: '10px', display: 'flex', margin: 'auto',
    background: 'red',
}

const PreviewCanvas = () => {


    return (
        <div style={centredStyle}>
            <canvas></canvas>
        </div>
    );
}

export default PreviewCanvas